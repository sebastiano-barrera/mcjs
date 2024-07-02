//
// mcjs's "bootstrap" script.
//
// This is a special script that is evaluated at the initialization of a
// mcjs interpreter (the Loader, to be precise). It initializes a subset of
// JavaScript built-ins that are hard or problematic to implement in native
// (Rust) code.
//
// Unlike other scripts, this one is allowed to use special syntax of the form
// $Instr(...) that is directly compiled to a specific bytecode instruction (or
// small set of instructions).


// TODO TODO TODO This needs to be updated to support more than 8 args
Function.prototype.call = function (new_this, arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
    // `this` is the function to call
    const bound = this.bind(new_this);
    return bound(arg0, arg1, arg2, arg3, arg4, arg5, arg6);
}

Function.prototype.apply = function (new_this, args) {
    // TODO change once spread syntax (e.g. `f(...args)`) is implemented
    // `this` is the function to call
    args = args || [];
    return this.bind(new_this)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
}

// TODO ensure function's name is fromCodePoint
String.fromCodePoint = function (codePoint) {
  if (codePoint === undefined) {
    return '';
  }

  codePoint = $ToNumber($ToPrimitive(codePoint));
  if ($NumberNotInteger(codePoint)) {
    throw new RangeError('invalid code point: ' + codePoint);
  }
  return $StrFromCodePoint(codePoint);
};


