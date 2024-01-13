require 'optparse'
require 'pathname'
require 'sqlite3'
require 'digest/sha1'

require_relative './common'

#
# Parsing flags
#

options = {}

OptionParser.new do |opt|
  opt.on('--mcjs ROOT', 'Base directory of the mcjs repo')
  opt.on('--test262 ROOT', 'Base directory of the test262 repo')
end.parse(ARGV, into: options)

if options[:test262].nil?
  abort "Option '--test262' is mandatory.  Try: scan_test262 --help"
end

#
# Main
#

mcjs_home = options[:mcjs] || Dir.pwd
test262_home = options[:test262]

SourceInst.new(repo_home: mcjs_home).scan_test262(test262_home)

