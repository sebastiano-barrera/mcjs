require 'pathname'
require 'json'
require 'digest/sha1'

require 'git'
require 'sqlite3'

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
# Database utils
#

Field = Struct.new(:type)

class TableSchema
  def initialize(name, fields)
    @name = name
    @fields = fields
  end

  attr_reader :name

  def fields()
    @fields.keys
  end

  def field_type field
    @fields[field].type
  end

  def field_name field
    raise Exception.new "no such field: #{field}" unless @fields.include? field
    field.to_s
  end
end

class Table
  def initialize(db, schema)
    @db = db
    @schema = schema
    @insert_stmt = nil
  end

  def delete_all
    @db.execute "delete from #{@schema.name}"
  end

  def recreate
    @db.execute "drop table if exists #{@schema.name}"

    field_descs = @schema.fields.map do |field|
      type = @schema.field_type field
      "#{field} #{type}"
    end
    @db.execute "create table #{@schema.name} (#{field_descs.join ', '})"
  end

  def insert record
    values = @schema.fields.map {|field| record[field.to_sym]}

    extra_attrs = Set.new(record.keys) - Set.new(@schema.fields) 
    raise Exception.new, "extra attributes: #{extra_attrs.join ', '}" unless extra_attrs.empty?

    @insert_stmt ||= @db.prepare "
      insert into #{@schema.name} (#{@schema.fields.join ', '})
      values (#{values.map{|_| '?'}.join ', '})
    "
    @insert_stmt.execute values
  end
end

#
# Main exports
#

TESTCASES = TableSchema.new 'testcases', {
  :path => Field.new(type: 'varchar'),
  :path_hash => Field.new(type: 'varchar'),
  :dirname => Field.new(type: 'varchar'),
  :basename => Field.new(type: 'varchar'),
  :uses_eval => Field.new(type: 'boolean'),
  :expected_error => Field.new(type: 'varchar'),
}

RUNS = TableSchema.new 'runs', {
  :path_hash => Field.new(type: 'blob'),
  :is_strict => Field.new(type: 'boolean'),
  :error_category => Field.new(type: 'varchar'),
  :error_message => Field.new(type: 'varchar'),
  :version => Field.new(type: 'varchar'),
}

class Database
  def initialize filename
    filename = Pathname.new filename
    new_file = ! filename.exist?
    FileUtils.mkdir_p(filename.dirname) 

    @db = SQLite3::Database.new filename
    @testcases = Table.new @db, TESTCASES
    @runs = Table.new @db, RUNS

    if new_file
      @db.transaction { self.recreate }
    end

    @stmt__delete_runs_for_version = nil
  end

  def transaction &block
    @db.transaction &block
  end

  def execute *args
    @db.execute *args
  end

  def delete_all_testcases
    @testcases.delete_all
  end

  def insert_testcase record
    @testcases.insert record
  end

  def delete_all_runs
    @runs.delete_all
  end

  def delete_runs_for_version version
    @stmt__delete_runs_for_version ||= @db.prepare "
      delete from #{RUNS.name}
      where #{RUNS.field_name :version} = ?
    "
    @stmt__delete_runs_for_version.execute(version.to_s)
  end

  def insert_run record
    @runs.insert record
  end

  def recreate_views
    @db.execute 'drop view if exists general'
    @db.execute '
      create view general as 
      select tc.path
      , r.version
      , r.error_message is null as success
      , r.is_strict
      , tc.dirname
      , tc.basename
      , tc.uses_eval 
      , tc.expected_error
      from runs r left join testcases tc on (r.path_hash = tc.path_hash)
    '

    @db.execute 'drop view if exists status'
    @db.execute '
      create view status as
      with q as (
      	select version
        , dirname
      	, success
      	, count(*) as count
      	from general
      	group by version, dirname, success 
      	order by version, dirname, success
      )
      , q2 as (
      	select version, dirname
      	, ifnull(sum(count) filter (where success = 1), 0) as ok
      	, ifnull(sum(count) filter (where success = 0), 0) as fail
      	from q
      	group by version, dirname
      )
      select *
      , cast(ok as real) * 100 / (ok + fail) as progress
      from q2
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

        puts path
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

  private

  def recreate
    STDERR.puts "initing schema"
    @runs.recreate
    @testcases.recreate
    self.recreate_views
  end
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
    @mcjs_version ||= @repo.log[0].sha
  end

  def transaction &block
    @db.transaction &block
  end

  def repo_home
    @repo.dir
  end

  def all_commits
    @repo.log.between(@first_commit, "HEAD")
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

  # Imports runs.json into database
  def import_outcome
    filename = self.runs_json_path

    STDERR.puts "loading #{filename}..."
    runs = File.new(filename).each_line.map{|line| JSON::load line}

    STDERR.puts "inserting into db..."
    @db.transaction do
      @db.delete_runs_for_version(self.mcjs_version)
      runs.each do |run|
        record = {
          :path_hash => Digest::SHA1.digest(run["file_path"]),
          :is_strict => run["is_strict"] ? 1 : 0,
          :error_category => run.deep_get("error", "category"),
          :error_message => run.deep_get("error", "message"),
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

