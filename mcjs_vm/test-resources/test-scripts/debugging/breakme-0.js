// The position of the breakpoint is hardcoded in the test case.
// If you change this file, also edit mcjs_vm/tests/debugging.rs

function foo() {
    sink(1);
    sink(2);
}

foo();
 
