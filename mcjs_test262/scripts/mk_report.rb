#!/usr/bin/env ruby

require 'optparse'
require 'git'
require 'sqlite3'

require_relative './common'

options = {}
OptionParser.new do |opt|
  opt.on('--mcjs-home ROOT', 'Directory of the mcjs repo. Default: pwd.')
  opt.on('--first-commit COMMIT', 'First commit to track')
  opt.on('--db FILENAME', 'Path to "tests.db". Default: ROOT/out/tests.db')
end.parse(ARGV, into: options)

inst = SourceInst.new(
  repo_home: options[:"mcjs-home"] || Dir.pwd,
  first_commit: options[:"first-commit"],
  db_filename: options[:db],
)

# list of SHAs
by_version = inst.status_by_version
pp by_version

inst.all_commits.each do |commit|
  h = by_version[commit.sha]
  headline = if h.nil?
    "(no data)"
  else
    percentage = h[:count_ok].to_f / h[:count_fail] * 100
    "%5d ok  %5d failed  %4.1f" % [
      h[:count_ok],
      h[:count_fail],
      percentage
    ]
  end

  puts "#{commit.sha}  #{headline}"
end


