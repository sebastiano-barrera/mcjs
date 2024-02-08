require_relative './common'

Dir.chdir __dir__

db = Database.new '../out/tests.db'
db.transaction do
  db.recreate_extras
end

