#!/usr/bin/env ruby

require_relative './common'

version_pre = ARGV.shift
version_post = ARGV.shift || "HEAD"

if version_pre.nil?
  abort "Usage: compare.rb <version pre> [version post]"
end

puts "comparing: #{version_pre} -> #{version_post}"
puts

inst = SourceInst.new(repo_home: __dir__)
new_successes = []
new_failures = []

inst.transaction do
  inst.outcome_diffs(version_pre, version_post).each do |item|
    if item.cur_success == 1
      new_successes << item
    else
      new_failures << item
    end
  end
end

puts "New successes (#{new_successes.length}):"
new_successes.each do |item|
  puts "  #{item.path} (#{item.is_strict ? "strict" : "non-strict"})"
end
puts

puts "New failures (#{new_failures.length}):"
new_failures.each do |item|
  puts "  #{item.path} (#{item.is_strict ? "strict" : "non-strict"})"
end
puts 

