// The position of the breakpoint is hardcoded in the test case.
// If you change this file, also edit mcjs_vm/tests/debugging.rs

function foo() {}

for (let i=0; i < 1_000_000; ++i) {
  foo();
}


