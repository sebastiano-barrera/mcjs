require 'pathname'
require 'json'
require 'digest/sha1'

require 'git'
require 'sqlite3'

require 'rspec'

class InsertStmtException < StandardError
end

class InsertStmt
  def initialize(db, table_name, column_names)
    @column_names = Set.new column_names

    columns = column_names.join ', '
    values_placeholders = column_names.map {|k| ":#{k}" }.join ", "
    @stmt = db.prepare "insert into #{table_name} (#{columns}) values (#{values_placeholders})"
  end

  def insert record
    if record.keys.to_set != @column_names
      raise InsertStmtException.new
    end
    
    begin
      @stmt.execute record
    rescue SQLite3::Exception
      raise InsertStmtException.new
    end
  end
end

RSpec.describe InsertStmt do
  before do
    @db = SQLite3::Database.new ":memory:"
    @db.execute 'create table strings (string varchar, hash varchar)'
  end

  it 'initializes with Symbol column names' do
    InsertStmt.new @db, 'strings', [:string, :hash]
  end
  it 'initializes with String column names' do
    InsertStmt.new @db, 'strings', ['string', 'hash']
  end

  it 'can insert a simple correct record' do
    stmt = InsertStmt.new @db, 'strings', [:string, :hash]
    stmt.insert({
      :string => "asdlol123",
      :hash => 'deadbeef',
    })

    res = @db.execute 'select string, hash from strings'
    expect(res).to eq([['asdlol123', 'deadbeef']])
  end

  it 'raises exception on extra attributes' do
    stmt = InsertStmt.new @db, 'strings', [:string, :hash]
    expect {
      stmt.insert({
        :string => "asdlol123",
        :hash => 'deadbeef',
        :something_else => 123,
      })
    }.to raise_error(InsertStmtException)
  end

  it 'raises exception on missing attributes' do
    stmt = InsertStmt.new @db, 'strings', [:string, :hash]
    expect {
      stmt.insert({
        :string => "asdlol123",
      })
    }.to raise_error(InsertStmtException)
  end
end


#
# Utils
#

HERE = File.expand_path __dir__

class Hash
  def deep_get(*keys)
    value = self
    keys.each do |k|
      break if value.nil?
      value = value[k]
    end
    value
  end
end

#
# Main exports
#

DiffItem = Struct.new(:path, :is_strict, :prev_success, :cur_success)

class Database
  def initialize filename
    if filename == ':memory:'
      new_file = false
    else
      filename = Pathname.new filename
      new_file = ! filename.exist?
      FileUtils.mkdir_p(filename.dirname) 
    end

    @db = SQLite3::Database.new filename

    if new_file
      @db.transaction { self.recreate }
    end
  end

  def transaction &block
    @db.transaction &block
  end

  def delete_all_testcases
    @db.execute 'delete from testcases'
  end

  def insert_testcase record
    @stmt__testcase_insert ||= InsertStmt.new @db, 'testcases', [
      :path,
      :path_hash,
      :dirname,
      :basename,
      :uses_eval,
      :expected_error,
    ]
    @stmt__testcase_insert.insert record
  end

  def delete_all_runs
    @db.execute 'delete from runs'
  end

  def delete_runs_for_version version
    @stmt__delete_runs_for_version ||= @db.prepare "delete from runs where version = ?"
    @stmt__delete_runs_for_version.execute(version.to_s)
  end

  def insert_run record
    @stmt__runs_insert ||= InsertStmt.new @db, 'runs', [
      :path_hash,
      :is_strict,
      :error_category,
      :error_message_hash,
      :version,
    ]
    @stmt__runs_insert.insert record
  end

  def insert_string string
    return nil if string.nil?
    hash = Digest::SHA1.hexdigest(string)
    # ignore if the string/hash pair is already in
    @db.execute 'insert or ignore into strings (string, hash) values (?, ?)', [string, hash]
    hash
  end

  def recreate_extras
    @db.execute 'drop view if exists general'
    @db.execute '
      create view general as 
      select tc.path
      , r.version
      , r.error_message_hash is null as success
      , s.string as error_message
      , r.is_strict
      , tc.dirname
      , tc.basename
      , tc.uses_eval 
      , tc.expected_error
      from runs r
        left join testcases tc on (r.path_hash = tc.path_hash)
        left join strings s on (r.error_message_hash = s.hash)
    '

    @db.execute '
      drop index if exists runs__version;        CREATE INDEX runs__version on runs (version);
      drop index if exists testcases__dirname;   CREATE INDEX testcases__dirname on testcases (dirname);
      drop index if exists testcases__path_hash; CREATE INDEX testcases__path_hash on testcases (path_hash);
    '
  end
    
  def scan_test262(root_dir)
    # Only scan under test/language/
    root_dir = Pathname.new(root_dir).join('test/language').cleanpath

    self.transaction do
      self.delete_all_testcases
      
      root_dir.glob('**/*.js') do |path|
        in_meta_block = true
        uses_eval = false
        expected_error = nil

        path.open.each_line do |line|
          if in_meta_block
            expected_error = $1 if /^\s+type: (\w+Error)/ =~ line
            in_meta_block = false if line =~ /---\*\//
          else
            uses_eval = true if not uses_eval and line =~ /\beval\b/
            in_meta_block = true if line =~ /\/\*---/
          end
        end

        record = {
          :path => path.to_s,
          :path_hash => Digest::SHA1.digest(path.to_s),
          :basename => path.basename.to_s,
          :dirname => path.dirname.to_s,
          :uses_eval => uses_eval ? 1 : 0,
          :expected_error => expected_error,
        }
        self.insert_testcase record
      end
    end
  end

  def outcome_diffs(version_pre, version_post)
    @stmt__outcome_diffs ||= @db.prepare '
      select cur.path
      , cur.is_strict
      , prev.success as prev_success
      , cur.success as cur_success
      from (
              select path
              , is_strict
              , success
              from general
              where version = ?
      ) as prev left join (
              select path
              , is_strict
              , success
              from general
              where version = ?
      ) as cur on (prev.path = cur.path and prev.is_strict = cur.is_strict)
      where prev_success <> cur_success
    '
    @stmt__outcome_diffs.execute([version_pre, version_post]).map do |row|
      path, is_strict, prev_success, cur_success = row
      DiffItem.new(
        path: path,
        is_strict: is_strict,
        prev_success: prev_success,
        cur_success: cur_success,
      )
    end
  end

  private

  def recreate
    STDERR.puts "initing schema"
    @runs.recreate
    @testcases.recreate
    @db.execute 'drop table if exists strings'
    @db.execute 'create table strings (string varchar, hash varchar, unique (string, hash))'
    self.recreate_extras
  end
end


RSpec.describe Database do
  it 'can initialize with :memory: as filename' do
    db = Database.new ':memory:'
    expect(Pathname.new(':memory:').exist?).to be true
  end

  it 
end


class SourceInst
  def initialize(repo_home:, db_filename: nil, first_commit: nil)
    @repo = Git::open(repo_home)

    db_filename ||= "#{@repo.dir}/mcjs_test262/out/tests.db"
    @db = Database.new(db_filename)

    @first_commit = first_commit || "32fb4783c60d1ffeb6db872600425fdf1e900225"
    @mcjs_version = nil
  end

  def mcjs_version
    if @mcjs_version.nil?
      status = @repo.status
      is_clean = status.changed.empty? \
        and status.added.empty? \
        and status.deleted.empty?
      @mcjs_version = is_clean ? @repo.log[0].sha : "dirty"
    end

    @mcjs_version
  end

  def transaction &block
    @db.transaction &block
  end

  def repo_home
    @repo.dir
  end

  def all_commits
    @repo.log(count = nil).between(@first_commit, "HEAD")
  end

  def tested_commits_set
    @db \
      .execute('select distinct version from status') \
      .flatten.to_set
  end

  def commits_to_test_shas
    tested = self.tested_commits_set
    self.all_commits \
      .map {|c| c.sha} \
      .filter{|sha| not tested.include? sha}
  end

  def checkout *args
    @repo.checkout *args
    # invalidate cache, will be updated on the next call to #mcjs_version
    @mcjs_version = nil 
  end

  # Run the test, generate out/runs.json, import into db
  def run
    # The run.sh script is taken from the repo, which is not necessarily the
    # sibling of this script.
    system "#{@repo.dir}/mcjs_test262/scripts/run.sh"
    self.import_outcome
  end

  def status_by_version
    records = @db.execute('select version, sum(ok) as ok, sum(fail) as fail from status group by version')

    by_version = {}
    records.each do |record|
      version, count_ok, count_fail = record
      abort "assertion error: duplicate version #{version}" \
          if by_version.include? version
      by_version[version] = {
        :count_ok => count_ok,
        :count_fail => count_fail
      }
    end

    by_version
  end

  def outcome_diffs(version_pre, version_post)
    version_pre = @repo.revparse version_pre unless version_pre == "dirty"
    version_post = @repo.revparse version_post unless version_post == "dirty"
    @db.outcome_diffs(version_pre, version_post)
  end

  # Imports runs.json into database
  def read_runs_json
    filename = self.runs_json_path
    STDERR.puts "loading #{filename}..."
    runs = File.new(filename).each_line.map{|line| JSON::load line}
  end

  def import_outcome
    runs = self.read_runs_json

    STDERR.puts "inserting into db..."
    @db.transaction do
      @db.delete_runs_for_version(self.mcjs_version)
      runs.each do |run|
        record = {
          :path_hash => Digest::SHA1.digest(run["file_path"]),
          :is_strict => run["is_strict"] ? 1 : 0,
          :error_category => run.deep_get("error", "category"),
          :error_message_hash => @db.insert_string(run.deep_get("error", "message")),
          :version => self.mcjs_version,
        }
        @db.insert_run record
      end
    end

    STDERR.puts "inserted #{runs.length} records"
  end

  def scan_test262 *args
    @db.scan_test262 *args
  end

  private
  def runs_json_path
    "#{@repo.dir}/mcjs_test262/out/runs.json"
  end
end

