require 'sqlite3'
require 'pathname'

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

TESTCASES = TableSchema.new 'testcases', {
  :path => Field.new(type: 'varchar'),
  :dirname => Field.new(type: 'varchar'),
  :basename => Field.new(type: 'varchar'),
  :uses_eval => Field.new(type: 'boolean'),
  :expected_error => Field.new(type: 'varchar'),
}

RUNS = TableSchema.new 'runs', {
  :path => Field.new(type: 'varchar'),
  :is_strict => Field.new(type: 'boolean'),
  :error_category => Field.new(type: 'varchar'),
  :error_message => Field.new(type: 'varchar'),
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
  end

  def transaction &block
    @db.transaction &block
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

  def insert_run record
    @runs.insert record
  end

  def recreate_views
    @db.execute 'drop view if exists general'
    @db.execute '
      create view general as 
      select r.path
      , r.error_message is null as success
      , tc.dirname
      , tc.basename
      , tc.uses_eval 
      , tc.expected_error
      from runs r left join testcases tc on (r.path = tc.path)
    '

    @db.execute 'drop view if exists status'
    @db.execute '
      create view status as
      with q as (
      	select dirname
      	, success
      	, count(*) as count
      	from general
      	group by dirname, success 
      	order by dirname, success
      )
      , q2 as (
      	select dirname
      	, ifnull(sum(count) filter (where success = 1), 0) as ok
      	, ifnull(sum(count) filter (where success = 0), 0) as fail
      	from q
      	group by dirname
      )
      select *
      , cast(ok as real) * 100 / (ok + fail) as progress
      from q2
    '
  end
    
  private

  def recreate
    STDERR.puts "initing schema"
    @runs.recreate
    @testcases.recreate
    self.recreate_views
  end
end


