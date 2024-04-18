package main

// TODOs:
//
// 	Features:
// 		Initialization
//			- [*] scan test262 JS source flies
//				// get expected error
//		Execution
//			- [*] run full test suite
//			- [*] import outcome file into db
//			- [ ] run specific test case
//			- [ ] run debugger on test case
//		Analysis
//			- [*] compare test results between VM versions
//			- [*] generate status page ("Are we ECMAScript yet?")
//			- [*] quick overview, broken down per directory
//		Extras
//			- [*] add a bunch of 'not null' to the schema

import (
	"bufio"
	"context"
	"crypto/sha1"
	"database/sql"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	_ "github.com/mattn/go-sqlite3"

	trag "github.com/sebastiano-barrera/mcjs/trag"
	tragdb "github.com/sebastiano-barrera/mcjs/trag/pkg/trag/db"
)

var (
	flagDBPath      *string = flag.String("db", "tests.db", "Path to the database")
	flagTest262Root *string = flag.String("test262", "", "Path to the test262 repository")

	flagInitEnabled       *bool   = flag.Bool("init", false, "Initialize database")
	flagInitTestsFilename *string = flag.String("initTests", "", "For -init: Test list file")

	flagRunEnabled  *bool   = flag.Bool("run", false, "Run test cases")
	flagRunFilter   *string = flag.String("runFilter", "", "Only run test cases whose path contains this string")
	flagRunNoRecord *bool   = flag.Bool("runNoRecord", false, "When passed, the test run is not recorded (no changes at all are made to the database)")

	flagChartEnabled          *bool   = flag.Bool("chart", false, "Generate the 'Are we ECMAscript yet?' page")
	flagChartTemplateFilename *string = flag.String("chartTemplate", "", "For -chart: Template file. By default, an embedded template is used.")
)

func main() {
	var err error

	flag.Parse()
	anySubcommand := false

	if *flagInitEnabled {
		if *flagInitTestsFilename == "" {
			log.Fatal("Test cases list file is required. Pass it with -tests.")
		}

		err = initDatabase(*flagDBPath, *flagInitTestsFilename)
		anySubcommand = true
	}

	if *flagRunEnabled {
		err = runFullSuite(*flagDBPath)
		anySubcommand = true
	}

	if *flagChartEnabled {
		templateHTML := defaultChartsTemplate

		if *flagChartTemplateFilename != "" {
			templateHTMLBytes, err := os.ReadFile(*flagChartTemplateFilename)
			if err != nil {
				log.Fatalf("can't open template file: %s: %s", *flagChartTemplateFilename, err)
			}

			templateHTML = string(templateHTMLBytes)
		}

		err = generateChart(templateHTML)
		anySubcommand = true
	}

	if !anySubcommand {
		log.Fatalf("no subcommand passed. check with -help.")
	}
	if err != nil {
		log.Fatal(err)
	}

}

func initDatabase(dbFilename, testListFilename string) error {
	if *flagTest262Root == "" {
		return fmt.Errorf("required flag not specified: -test262")
	}

	testListFile, err := os.Open(testListFilename)
	if err != nil {
		return fmt.Errorf("open test list file %s: %w", testListFilename, err)
	}
	testListScnr := bufio.NewScanner(testListFile)

	db, err := sql.Open("sqlite3", dbFilename)
	if err != nil {
		return fmt.Errorf("open db %s: %w", dbFilename, err)
	}
	defer db.Close()

	ctx := context.TODO()
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("while starting transaction: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.ExecContext(ctx, trag.DatabaseDDL)
	if err != nil {
		return fmt.Errorf("while initializing schema: %w", err)
	}

	queries := tragdb.New(db).WithTx(tx)
	err = queries.ClearTestCases(ctx)
	if err != nil {
		return fmt.Errorf("while clearing existing data: %w", err)
	}

	for testListScnr.Scan() {
		// testCaseFilename is relative to the test262 root
		testCaseFilename := testListScnr.Text()
		testCaseFilenameHHex := internString(ctx, queries, testCaseFilename)

		// groups are labeled by their directiories
		groupName := path.Dir(testCaseFilename)
		groupNameHHex := internString(ctx, queries, groupName)

		// from the file, we get the path relative to the test262 root
		fullFilename := path.Join(*flagTest262Root, testCaseFilename)
		meta, err := scanTestCase(fullFilename)
		if err != nil {
			return fmt.Errorf("scanning test case %s: %w", testCaseFilename, err)
		}

		err = queries.InsertTestCase(ctx, tragdb.InsertTestCaseParams{
			PathHash:      testCaseFilenameHHex,
			ExpectedError: sql.NullString{String: meta.expectedError, Valid: true},
		})
		if err != nil {
			return fmt.Errorf("inserting test case: %w", err)
		}

		err = queries.AssignGroup(ctx, tragdb.AssignGroupParams{
			PathHash:  testCaseFilenameHHex,
			GroupHash: groupNameHHex,
		})
		if err != nil {
			return fmt.Errorf("assigning group name: %w", err)
		}
	}

	err = testListScnr.Err()
	if err != nil {
		return fmt.Errorf("test list read: %w", err)
	}

	err = tx.Commit()
	if err != nil {
		return fmt.Errorf("committing transaction: %w", err)
	}

	return nil
}

type testCaseMetadata struct {
	expectedError string
}

func scanTestCase(testCaseFilename string) (meta testCaseMetadata, err error) {
	testCaseFile, err := os.Open(testCaseFilename)
	if err != nil {
		err = fmt.Errorf("open test case %s: %w", testCaseFilename, err)
		return
	}
	testCaseScnr := bufio.NewScanner(testCaseFile)

	inMetaBlock := false

	for testCaseScnr.Scan() {
		line := testCaseScnr.Text()

		if inMetaBlock {
			if line == "---*/" {
				inMetaBlock = false
			} else {
				fmt.Sscanf(line, " type: %s ", &meta.expectedError)
			}
		} else if line == "/*---" {
			inMetaBlock = true
		}
	}

	err = testCaseScnr.Err()
	if err != nil {
		return testCaseMetadata{}, err
	}

	return meta, nil
}

func internString(ctx context.Context, queries *tragdb.Queries, s string) string {
	hash := sha1.Sum([]byte(s))
	hashHex := hex.EncodeToString(hash[:])
	queries.InsertString(ctx, tragdb.InsertStringParams{
		String: s,
		Hash:   hashHex,
	})
	return hashHex
}

func runFullSuite(dbFilename string) error {
	if *flagTest262Root == "" {
		return fmt.Errorf("required flag not passed: -test262")
	}

	// open db
	db, err := sql.Open("sqlite3", dbFilename)
	if err != nil {
		return fmt.Errorf("open db %s: %w", dbFilename, err)
	}
	defer db.Close()

	ctx := context.TODO()
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	queries := tragdb.New(db).WithTx(tx)

	// get list of test cases from db, generate configuration file
	relPaths, err := queries.ListTestCases(ctx)
	if err != nil {
		return fmt.Errorf("listing test cases: %w", err)
	}

	type runnerConfig struct {
		Test262Root string   `json:"test262Root"`
		TestFiles   []string `json:"testFiles"`
	}

	config := runnerConfig{
		Test262Root: *flagTest262Root,
		TestFiles:   make([]string, 0, len(relPaths)),
	}
	for _, relPath := range relPaths {
		if !relPath.Valid {
			panic("this query should not have returned a null!")
		}
		if !strings.Contains(relPath.String, *flagRunFilter) {
			continue
		}
		config.TestFiles = append(config.TestFiles, relPath.String)
	}

	configFile, err := os.CreateTemp("", "mcjs_test262.json-*")
	if err != nil {
		return fmt.Errorf("creating tmp config file: %w", err)
	}

	enc := json.NewEncoder(configFile)
	enc.Encode(config)
	configFile.Close()

	// check build version
	vmVersionBytes, err := exec.Command("cargo", "run", "-p", "mcjs_test262", "--", "--version").Output()
	if err != nil {
		return fmt.Errorf("test runner: failed to get build version: %s", err)
	}
	vmVersion := string(vmVersionBytes)
	vmVersion = strings.TrimSpace(vmVersion)

	if !*flagRunNoRecord {
		// delete runs for the same version
		// (cool that we can 'restore' them by rolling the transaction back,
		// which automatically happens on error)
		err = queries.DeleteRunsForVersion(
			ctx,
			vmVersion,
		)
		if err != nil {
			return fmt.Errorf("failed to delete previously recorded runs for version %s: %w", vmVersion, err)
		}
	}

	// run test runner
	cmd := exec.Command(
		"cargo",
		"run",
		"-p",
		"mcjs_test262",
		"--",
		configFile.Name(),
	)
	defer func() {
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
	}()
	log.Println("running command:", cmd)

	cmdOutput, err := cmd.StdoutPipe()
	if err != nil {
		err = fmt.Errorf("test runner: can't open stdout: %s", err)
		return err
	}

	err = cmd.Start()
	if err != nil {
		err = fmt.Errorf("test runner failed to start: %s", err)
		return err
	}

	cmdOutputScnr := bufio.NewScanner(cmdOutput)
	// pre-allocate a fixed LARGE buffer (4 MB)
	// the one-line-per-JSON format can produce some pretty large lines
	cmdOutputScnr.Buffer(make([]byte, 0, 4*1024*1024), 0)
	nInserts := 0
	for cmdOutputScnr.Scan() {
		line := cmdOutputScnr.Text()
		log.Println("output line:", line)

		if *flagRunNoRecord {
			break
		}

		var logItem struct {
			FullPath string `json:"file_path"`
			Error    *struct {
				Category, Message string
			}
			IsStrict bool `json:"is_strict"`
		}
		err = json.Unmarshal([]byte(line), &logItem)
		if err != nil {
			log.Printf("discarding line, can't parse as JSON: %s (line: %v)", err, line)
			continue
		}

		// test runner emits full path to test case, while db `runs`
		// record needs to match record in `testcases`, where path is
		// relative to test262 root.
		relPath, err := filepath.Rel(config.Test262Root, logItem.FullPath)
		if err != nil {
			log.Printf("skipping invalid test case path: %s: %s", logItem.FullPath, err)
			continue
		}

		relPathHHex := internString(ctx, queries, relPath)

		record := tragdb.InsertRunParams{
			PathHash: relPathHHex,
			Version:  vmVersion,
			IsStrict: logItem.IsStrict,
		}
		if logItem.Error != nil {
			record.ErrorCategory = sql.NullString{String: logItem.Error.Category, Valid: true}
			messageHHex := internString(ctx, queries, logItem.Error.Message)
			record.ErrorMessageHash = sql.NullString{String: messageHHex, Valid: true}
		}

		err = queries.InsertRun(ctx, record)
		if err != nil {
			// no point in continuing: if the db is broken somehow,
			// we're not going to be able to record and use the
			// test runner's output
			log.Printf("bailing out test run, due to error in inserting into DB")
			// subprocess will get killed in the defer block
			return fmt.Errorf("error while inserting in DB: %w", err)
		}
		nInserts++
	}

	readError := cmdOutputScnr.Err()
	if readError != nil {
		// error => fuck the process, but let's still commit the
		// partial data that we managed to collect (the next run can
		// overwrite it).
		readError = fmt.Errorf("test runner: read: %s\n", readError)
	}

	if !*flagRunNoRecord {
		err = tx.Commit()
		if err != nil {
			return fmt.Errorf("error while committing transaction (data will be discarded!): %s", err)
		}
	}

	log.Printf("work finished. transaction committed with %d inserts", nInserts)
	return readError
}

type commit struct {
	CommitID, Message        string
	CountPassed, CountFailed int64
	PercentPassedText        string
}

//go:embed charts.html
var defaultChartsTemplate string

func generateChart(templateHTML string) error {
	tmpl, err := template.New("page").Parse(templateHTML)
	if err != nil {
		return fmt.Errorf("parsing template: %w", err)
	}

	commits, err := listCommits()
	if err != nil {
		return fmt.Errorf("listing relevant commits: %w", err)
	}

	// open db
	dbFilename := *flagDBPath
	db, err := sql.Open("sqlite3", dbFilename)
	if err != nil {
		return fmt.Errorf("open db %s: %w", dbFilename, err)
	}
	defer db.Close()

	ctx := context.TODO()
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	byHash := make(map[string]*commit)
	for i := range commits {
		byHash[commits[i].CommitID] = &commits[i]
	}

	queries := tragdb.New(db).WithTx(tx)
	successes, err := queries.CountSuccessesByVersion(ctx)
	if err != nil {
		return fmt.Errorf("query error: %w", err)
	}

	failures, err := queries.CountFailuresByVersion(ctx)
	if err != nil {
		return fmt.Errorf("query error: %w", err)
	}

	for _, success := range successes {
		commit, ok := byHash[success.Version]
		if !ok {
			continue
		}
		commit.CountPassed = success.Count
	}

	for _, failure := range failures {
		commit, ok := byHash[failure.Version]
		if !ok {
			continue
		}
		commit.CountFailed = failure.Count
	}

	filteredCommits := make([]commit, 0, len(commits))
	for i := range commits {
		c := &commits[i]
		total := c.CountPassed + c.CountFailed
		percentPassed := float32(c.CountPassed) / float32(total) * 100
		c.PercentPassedText = fmt.Sprintf("%.1f%%", percentPassed)

		if total > 0 {
			filteredCommits = append(filteredCommits, commits[i])
		}
	}

	err = tmpl.ExecuteTemplate(os.Stdout, "page", struct{ Commits []commit }{
		Commits: filteredCommits,
	})
	if err != nil {
		return fmt.Errorf("rendering template: %w", err)
	}

	return nil
}

func listCommits() ([]commit, error) {
	var commits []commit

	// collect commits to show
	// the first commit (32fb4783) is hardcoded, and it's just the first
	// commit where I started running tests and collecting results
	cmd := exec.Command("git", "log", "--format=%H|%s", "32fb4783c60d1ffeb6db872600425fdf1e900225..")
	cmdOutput, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("couldn't get output pipe for `git log` command: %w", err)
	}
	defer cmdOutput.Close()
	err = cmd.Start()
	if err != nil {
		return nil, fmt.Errorf("couldn't list commits (`git log` command): %w", err)
	}

	scnr := bufio.NewScanner(cmdOutput)
	for scnr.Scan() {
		line := scnr.Text()
		sepNdx := strings.IndexRune(line, '|')
		if sepNdx == -1 {
			log.Printf("warning: skipping invalid line: %s", line)
			continue
		}

		commits = append(commits, commit{
			CommitID: line[0:sepNdx],
			Message:  line[sepNdx+1:],
		})
	}

	err = scnr.Err()
	if err != nil {
		return nil, fmt.Errorf("reading error: %w", err)
	}
	return commits, nil
}