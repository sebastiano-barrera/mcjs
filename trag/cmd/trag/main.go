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
//			- [ ] compare test results between VM versions
//			- [ ] generate status page ("Are we ECMAScript yet?")
//			- [ ] quick overview, broken down per directory

import (
	"bufio"
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path"
	"strings"

	"database/sql"

	_ "github.com/mattn/go-sqlite3"

	trag "github.com/sebastiano-barrera/mcjs/trag"
	tragdb "github.com/sebastiano-barrera/mcjs/trag/pkg/trag/db"
)

var (
	flagDBPath      *string
	flagTest262Root *string

	initFS            *flag.FlagSet
	initTestsFilename *string

	subcommands map[string]func([]string)
)

func init() {
	flagDBPath = flag.String("db", "tests.db", "Path to the database")
	flagTest262Root = flag.String("test262", "", "Path to the test262 repository")

	initFS = flag.NewFlagSet("init", flag.ExitOnError)
	initTestsFilename = initFS.String("tests", "", "Test list file.")

	subcommands = map[string]func([]string){
		"init": mainInit,
		"run":  mainRunFull,
	}
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "-help" {
		mainHelp()
		return
	}

	flag.Parse()
	args := flag.Args()

	if len(args) == 0 {
		log.Fatalf("no subcommand passed. check with -help.")
	}

	handler, ok := subcommands[args[0]]
	if ok {
		handler(args[1:])
		return
	}

	log.Fatalf("no such subcommand `%s`. check usage with -help.", os.Args[1])
}

func mainHelp() {
	flag.Usage()

	fmt.Print("subcommands: ")
	first := true
	for cmd := range subcommands {
		if !first {
			fmt.Print(", ")
		}
		fmt.Printf("%s", cmd)
		first = false
	}
	fmt.Println()
}

func mainInit(args []string) {
	initFS.Parse(args)

	if *initTestsFilename == "" {
		log.Fatal("Test cases list file is required. Pass it with -tests.")
	}

	err := initDatabase(*flagDBPath, *initTestsFilename)
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
		testCaseFilename := testListScnr.Text()

		// from the file, we get the path relative to the test262 root
		fullFilename := path.Join(*flagTest262Root, testCaseFilename)

		meta, err := scanTestCase(fullFilename)
		if err != nil {
			return fmt.Errorf("scanning test case %s: %w", testCaseFilename, err)
		}

		testCaseFilenameHB64 := internString(ctx, queries, testCaseFilename)
		err = queries.InsertTestCase(ctx, tragdb.InsertTestCaseParams{
			PathHash:      sql.NullString{String: testCaseFilenameHB64, Valid: true},
			ExpectedError: sql.NullString{String: meta.expectedError, Valid: true},
		})
		if err != nil {
			return fmt.Errorf("inserting test case: %w", err)
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
	hashBase64 := hex.EncodeToString(hash[:])
	queries.InsertString(ctx, tragdb.InsertStringParams{
		String: sql.NullString{String: s, Valid: true},
		Hash:   sql.NullString{String: hashBase64, Valid: true},
	})
	return hashBase64
}

func mainRunFull(args []string) {
	if len(args) != 0 {
		log.Fatalf("no command line args supported yet")
	}

	err := runFullSuite(*flagDBPath)
	if err != nil {
		log.Fatal(err)
	}
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
		TestFiles:   make([]string, len(relPaths)),
	}
	for i, relPath := range relPaths {
		if !relPath.Valid {
			panic("this query should not have returned a null!")
		}
		config.TestFiles[i] = relPath.String
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

	// delete runs for the same version
	// (cool that we can 'restore' them by rolling the transaction back,
	// which automatically happens on error)
	err = queries.DeleteRunsForVersion(
		ctx,
		sql.NullString{String: vmVersion, Valid: true},
	)
	if err != nil {
		return fmt.Errorf("failed to delete previously recorded runs for version %s: %w", vmVersion, err)
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
	for cmdOutputScnr.Scan() {
		line := cmdOutputScnr.Text()
		log.Println("output line:", line)

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
		relPath, hasPrefix := strings.CutPrefix(logItem.FullPath, config.Test262Root)
		if !hasPrefix {
			log.Printf("discarding line, test case path is not under test262 root path: %s", logItem.FullPath)
			continue
		}

		relPathHB64 := internString(ctx, queries, relPath)

		record := tragdb.InsertRunParams{
			PathHash: sql.NullString{String: relPathHB64, Valid: true},
			Version:  sql.NullString{String: vmVersion, Valid: true},
			IsStrict: sql.NullBool{Bool: logItem.IsStrict, Valid: true},
		}
		if logItem.Error != nil {
			record.ErrorCategory = sql.NullString{String: logItem.Error.Category, Valid: true}
			messageHB64 := internString(ctx, queries, logItem.Error.Message)
			record.ErrorMessageHash = sql.NullString{String: messageHB64, Valid: true}
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
	}

	readError := cmdOutputScnr.Err()
	if readError != nil {
		// don't return now, just wrap it for later
		readError = fmt.Errorf("test runner: read: %s\n", err)
	}

	err = cmd.Wait()
	if err != nil {
		return fmt.Errorf("test runner failed: %s", err)
	}

	err = tx.Commit()
	if err != nil {
		return fmt.Errorf("error while committing transaction (data will be discarded!): %s", err)
	}

	if readError != nil {
		return readError
	}
	return nil
}
