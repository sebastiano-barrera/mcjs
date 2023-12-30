require 'optparse'
require 'json'

require './common'

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
# Main
#

Dir.chdir __dir__

filename = '../out/runs.json'
STDERR.puts "loading #{filename}..."
runs = File.new(filename).each_line.map{|line| JSON::load line}

STDERR.puts "inserting into db..."
db = Database.new '../out/tests.db'
db.transaction do
  db.delete_all_runs
  runs.each do |run|
    record = {
      :path => run["file_path"],
      :is_strict => run["is_strict"] ? 1 : 0,
      :error_category => run.deep_get("error", "category"),
      :error_message => run.deep_get("error", "message"),
    }
    puts record
    db.insert_run record
  end
end

STDERR.puts "inserted #{runs.length} records"


