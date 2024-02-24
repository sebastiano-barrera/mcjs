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

pp options

inst = SourceInst.new(
  repo_home: options[:"mcjs-home"] || Dir.pwd,
  first_commit: options[:"first-commit"],
  db_filename: options[:db],
)

init_version = inst.mcjs_version

begin
  commits = inst.commits_to_test_shas
  puts "commits to test (#{commits.length}):"
  commits.each do |sha|
    puts "COMMIT #{sha}"
    inst.checkout sha
    inst.run
    abort "assertion failed: hash is #{inst.mcjs_version}, instead of the expected #{sha}" \
      if inst.mcjs_version != sha
  end
ensure
  commits = inst.commits_to_test_shas
  puts "finished.  remaining commits to test (#{commits.length}):"
  commits.each do |sha|
    puts "  #{sha}"
  end

  puts "returning to #{init_version}"
  inst.checkout init_version
end


