require 'optparse'
require 'pathname'
require 'sqlite3'

require './common'

Dir.chdir __dir__

#
# Parsing flags
#

options = {}

OptionParser.new do |opt|
  opt.on('--root ROOT', 'Base directory of the test262 repo')
end.parse(ARGV, into: options)

if options[:root].nil?
  abort "Option '--root' is mandatory.  Try: scan_test262 --help"
end

root_dir = Pathname.new(options[:root]).join('test/language').cleanpath

#
# Main
#

db = Database.new '../out/tests.db'
db.transaction do
  db.delete_all_testcases
  
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
      :basename => path.basename.to_s,
      :dirname => path.dirname.to_s,
      :uses_eval => uses_eval ? 1 : 0,
      :expected_error => expected_error,
    }
    db.insert_testcase record
  end
end


