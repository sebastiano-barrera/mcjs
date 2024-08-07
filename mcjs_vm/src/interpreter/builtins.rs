#![allow(non_snake_case)]

use std::rc::Rc;

use super::{
    make_exception, to_boolean, to_number, to_object, to_object_or_throw, Closure, NativeFn,
    RunError, RunResult,
};
use super::{to_string_or_throw, Realm, Value};

use crate::error;
use crate::heap;
use crate::heap::JSString;
use crate::heap::Property;
use crate::loader;

#[inline]
fn native_closure(nf: NativeFn) -> Rc<Closure> {
    Rc::new(Closure::new_native(nf))
}

pub(super) fn init_builtins(heap: &mut heap::Heap) -> Value {
    let Array = {
        let Array_push = Value::Object(heap.new_function(native_closure(nf_Array_push)));
        let Array_pop = Value::Object(heap.new_function(native_closure(nf_Array_pop)));
        let array_proto = Value::Object(heap.array_proto());
        {
            heap.set_own(
                array_proto,
                "push".into(),
                Property::NonEnumerable(Array_push),
            );
            heap.set_own(
                array_proto,
                "pop".into(),
                Property::NonEnumerable(Array_pop),
            );
        }

        let Array_isArray = Value::Object(heap.new_function(native_closure(nf_Array_isArray)));
        let array_ctor = Value::Object(heap.new_function(native_closure(nf_Array)));
        {
            heap.set_own(
                array_ctor,
                "isArray".into(),
                Property::NonEnumerable(Array_isArray),
            );
            heap.set_own(
                array_ctor,
                "prototype".into(),
                Property::NonEnumerable(array_proto),
            );
        }

        array_ctor
    };

    let RegExp_proto = Value::Object(heap.new_function(native_closure(nf_RegExp)));
    {
        let test = Value::Object(heap.new_function(native_closure(nf_RegExp_test)));
        heap.set_own(
            RegExp_proto,
            heap::IndexOrKey::Key("test"),
            Property::NonEnumerable(test),
        );
    }
    let RegExp = Value::Object(heap.new_function(native_closure(nf_RegExp)));
    heap.set_own(
        RegExp,
        heap::IndexOrKey::Key("prototype"),
        Property::NonEnumerable(RegExp_proto),
    );

    let Number = Value::Object(heap.new_function(native_closure(nf_Number)));
    {
        let number_proto = Value::Object(heap.number_proto());
        heap.set_own(
            Number,
            "prototype".into(),
            Property::NonEnumerable(number_proto),
        );

        let toString =
            Value::Object(heap.new_function(native_closure(nf_Number_prototype_toString)));
        heap.set_own(
            number_proto,
            "toString".into(),
            Property::NonEnumerable(toString),
        );

        let valueOf = Value::Object(heap.new_function(native_closure(nf_Number_prototype_valueOf)));
        heap.set_own(
            number_proto,
            "valueOf".into(),
            Property::NonEnumerable(valueOf),
        );
    }

    let String = Value::Object(heap.new_function(native_closure(nf_String)));

    {
        let string_proto = Value::Object(heap.string_proto());
        heap.set_own(String, "prototype".into(), {
            Property::NonEnumerable(string_proto)
        });

        let valueOf = Property::NonEnumerable(Value::Object(
            heap.new_native_function(nf_String_prototype_valueOf),
        ));
        heap.set_own(string_proto, heap::IndexOrKey::Key("valueOf"), valueOf);

        let toString = Property::NonEnumerable(Value::Object(
            heap.new_native_function(nf_String_prototype_toString),
        ));
        heap.set_own(string_proto, heap::IndexOrKey::Key("toString"), toString);

        let codePointAt = Property::NonEnumerable(Value::Object(
            heap.new_function(native_closure(nf_String_codePointAt)),
        ));
        heap.set_own(
            string_proto,
            heap::IndexOrKey::Key("codePointAt"),
            codePointAt,
        );
    }

    let Boolean = Value::Object(heap.new_function(native_closure(nf_Boolean)));
    {
        heap.set_own(Boolean, "prototype".into(), {
            let value = Value::Object(heap.bool_proto());
            Property::NonEnumerable(value)
        });
    }

    {
        let bool_proto = Value::Object(heap.bool_proto());
        let valueOf = Property::NonEnumerable(Value::Object(
            heap.new_native_function(nf_Boolean_prototype_valueOf),
        ));
        heap.set_own(bool_proto, heap::IndexOrKey::Key("valueOf"), valueOf);

        let toString = Property::NonEnumerable(Value::Object(
            heap.new_native_function(nf_Boolean_prototype_toString),
        ));
        heap.set_own(bool_proto, heap::IndexOrKey::Key("toString"), toString);
    }

    let Function = Value::Object(heap.new_function(native_closure(nf_Function)));
    {
        heap.set_own(Function, "prototype".into(), {
            let value = Value::Object(heap.func_proto());
            Property::NonEnumerable(value)
        });
    }

    {
        let func_proto = Value::Object(heap.func_proto());
        let toString = Property::NonEnumerable(Value::Object(
            heap.new_native_function(nf_Function_prototype_toString),
        ));
        heap.set_own(func_proto, heap::IndexOrKey::Key("toString"), toString);
    }

    let Object = Value::Object(heap.new_function(native_closure(nf_Object)));
    {
        heap.set_own(Object, "prototype".into(), {
            let value = Value::Object(heap.object_proto());
            Property::NonEnumerable(value)
        });
    }

    {
        let object_proto = Value::Object(heap.object_proto());

        let valueOf = Value::Object(heap.new_function(native_closure(nf_Object_prototype_valueOf)));
        heap.set_own(
            object_proto,
            "valueOf".into(),
            Property::NonEnumerable(valueOf),
        );

        let toString =
            Value::Object(heap.new_function(native_closure(nf_Object_prototype_toString)));
        heap.set_own(
            object_proto,
            "toString".into(),
            Property::NonEnumerable(toString),
        );
    }

    let cash_print = Value::Object(heap.new_function(native_closure(nf_cash_print)));

    let func_bind = Value::Object(heap.new_function(native_closure(nf_Function_bind)));
    {
        heap.set_own(
            Value::Object(heap.func_proto()),
            "bind".into(),
            Property::NonEnumerable(func_bind),
        );
    }

    // TODO(big feat) pls impl all Node.js API, ok? thxbye

    let global = Value::Object(heap.new_ordinary_object());
    heap.set_own(global, "Object".into(), Property::Enumerable(Object));
    heap.set_own(global, "Array".into(), Property::Enumerable(Array));
    heap.set_own(global, "RegExp".into(), Property::Enumerable(RegExp));
    heap.set_own(global, "Number".into(), Property::Enumerable(Number));
    heap.set_own(global, "String".into(), Property::Enumerable(String));
    heap.set_own(global, "Boolean".into(), Property::Enumerable(Boolean));
    heap.set_own(global, "Function".into(), Property::Enumerable(Function));
    heap.set_own(global, "$print".into(), Property::Enumerable(cash_print));

    let Error = add_exception_type(heap, "Error", Value::Object(heap.object_proto()), global);
    add_exception_type(heap, "ReferenceError", Error, global);
    add_exception_type(heap, "TypeError", Error, global);
    add_exception_type(heap, "SyntaxError", Error, global);
    add_exception_type(heap, "RangeError", Error, global);

    global
}

/// Create a new exception type.
///
/// The constructor will be named like `name`, and the new prototype will have
/// `parent_prototype` as its own prototype. The constructor will also be
/// assigned as a property into `container`.
fn add_exception_type(
    heap: &mut heap::Heap,
    name: &str,
    parent_prototype: Value,
    container: Value,
) -> Value {
    let parent_prototype = match parent_prototype {
        Value::Object(oid) => oid,
        _ => panic!("bug: add_exception_type: parent prototype must be object"),
    };

    let name_obj = Value::String(heap.new_string(JSString::new_from_str(name)));

    let prototype = Value::Object(heap.new_ordinary_object());
    heap.set_proto(prototype, Some(parent_prototype));
    heap.set_own(
        prototype,
        heap::IndexOrKey::Key("name"),
        Property::Enumerable(name_obj),
    );

    let cons = Value::Object(heap.new_function(native_closure(nf_set_message)));
    heap.set_own(
        cons,
        heap::IndexOrKey::Key("prototype"),
        Property::NonEnumerable(prototype),
    );

    heap.set_own(
        container,
        heap::IndexOrKey::Key(name),
        Property::Enumerable(cons),
    );

    prototype
}

fn nf_set_message(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    let message = *args.get(0).unwrap_or(&Value::Undefined);
    realm
        .heap
        .set_own(*this, "message".into(), Property::Enumerable(message));
    Ok(Value::Undefined)
}

fn nf_Array_isArray(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    _this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    let value = if let Some(obj) = args.first() {
        realm.heap.array_elements(*obj).is_some()
    } else {
        false
    };

    Ok(Value::Bool(value))
}

fn nf_Array_push(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    // TODO Proper error handling, instead of these unwrap
    let value = *args.first().unwrap();
    realm.heap.array_push(*this, value);
    Ok(Value::Undefined)
}

fn nf_Array_pop(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    _args: &[Value],
) -> RunResult<Value> {
    match realm.heap.as_array_mut(*this) {
        Some(elements) => Ok(elements.pop().unwrap_or(Value::Undefined)),
        None => {
            let exc = super::make_exception(realm, "Error", "not an array");
            Err(RunError::Exception(exc))
        }
    }
}

fn nf_RegExp(
    realm: &mut Realm,
    loader: &mut loader::Loader,
    this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    let source = match args.first() {
        Some(val) => to_string_or_throw(*val, realm, loader)?,
        None => {
            let jss = JSString::new_from_str(":?");
            let sid = realm.heap.new_string(jss);
            Value::String(sid)
        }
    };

    realm.heap.set_own(
        *this,
        heap::IndexOrKey::Key("source"),
        Property::NonEnumerable(source),
    );
    Ok(*this)
}

fn nf_RegExp_test(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    // TODO Insanely slow! Cache the Regex object

    let source = realm
        .heap
        .get_chained(*this, heap::IndexOrKey::Key("source"));
    let source = match source {
        Some(property) => property.value().unwrap(),
        None => return Ok(Value::Undefined),
    };
    let source = realm.heap.as_str(source).cloned().ok_or_else(|| {
        let exc = super::make_exception(realm, "Error", "property `source` is not a string");
        RunError::Exception(exc)
    })?;

    // TODO Avoid conversion of pattern from UTF-16
    let source = source.to_string();

    let regex = regress::Regex::new(&source).map_err(|re_err| {
        let msg = &format!("Regex error: {}", re_err);
        let exc = super::make_exception(realm, "Error", msg);
        RunError::Exception(exc)
    })?;

    let haystack = *args.first().unwrap_or(&Value::Undefined);
    let haystack = match realm.heap.as_str(haystack) {
        Some(s) => s,
        None => return Ok(Value::Bool(false)),
    };

    let found = regex.find_from_utf16(haystack.view(), 0).next().is_some();
    Ok(Value::Bool(found))
}

fn nf_Array(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    let this = this
        .expect_obj()
        .expect("compiler bug: new Array(...): non-object this");
    // args directly go as values
    realm.heap.init_array(this, args.to_vec());
    Ok(Value::Undefined)
}

fn nf_Number(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    _this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    let value = args.first().copied().unwrap_or(Value::Undefined);

    match value {
        Value::Number(_) => Ok(value),
        Value::Bool(true) => Ok(Value::Number(1.)),
        Value::Bool(false) => Ok(Value::Number(0.)),
        Value::Object(_) => Ok(Value::Number(f64::NAN)),
        Value::String(_) => {
            let value: f64 = realm
                .heap
                .as_str(value)
                .and_then(|jss| jss.to_string().parse::<f64>().ok())
                .unwrap_or(f64::NAN);

            Ok(Value::Number(value))
        }
        Value::Null => Ok(Value::Number(0.)),
        Value::Undefined => Ok(Value::Number(f64::NAN)),
        Value::Symbol(_) => {
            let message = "Cannot convert a Symbol value to a number";
            let exc = make_exception(realm, "TypeError", message);
            Err(RunError::Exception(exc))
        }
    }
}

fn nf_Number_prototype_valueOf(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    _args: &[Value],
) -> RunResult<Value> {
    realm
        .heap
        .as_primitive(*this)
        .filter(|x| matches!(x, Value::Number(_)))
        .ok_or_else(|| {
            let message = "Number.prototype.valueOf requires that 'this' be a Number";
            let exc = make_exception(realm, "TypeError", message);
            RunError::Exception(exc)
        })
}

fn nf_Number_prototype_toString(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    _args: &[Value],
) -> RunResult<Value> {
    let prim = realm.heap.as_primitive(*this).ok_or_else(|| {
        let msg = "not a primitive! (and therefore not a number)";
        let exc = super::make_exception(realm, "TypeError", msg);
        RunError::Exception(exc)
    })?;

    match prim {
        Value::Number(n) => {
            let jss = JSString::new_from_str(&n.to_string());
            let sid = realm.heap.new_string(jss);
            Ok(Value::String(sid))
        }
        _ => {
            let exc = super::make_exception(realm, "TypeError", "not a number");
            Err(RunError::Exception(exc))
        }
    }
}

fn nf_String(
    realm: &mut Realm,
    loader: &mut loader::Loader,
    this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    let value = args.first().copied();

    let value_str = match value {
        None => {
            let sid = realm.heap.new_string(JSString::empty());
            Value::String(sid)
        }
        Some(v) => to_string_or_throw(v, realm, loader)?,
    };
    debug_assert!(matches!(value_str, Value::String(_)));

    match this {
        Value::Object(_) => {
            // called as a constructor -> create primitive-wrapper
            // TODO discarding this. does this make sense?
            Ok(Value::Object(realm.heap.wrap_primitive(value_str)))
        }
        Value::Undefined => {
            // called as function, not constructor -> return string primitive
            Ok(value_str)
        }
        _ => panic!("bug: 'this' not an object: {:?}", this),
    }
}

fn nf_String_codePointAt(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    let index = {
        let index = args.first().copied().unwrap_or(Value::Number(0.0));
        let index = to_number(index, realm)?.ok_or_else(|| -> RunError {
            error!("invalid index: {}", realm.heap.show_debug(index)).into()
        })?;
        if index.fract() != 0.0 {
            return Err(error!("invalid index (not an integer): {}", index).into());
        }
        index as usize
    };

    let this_str = realm.heap.as_str(*this).ok_or_else(|| -> RunError {
        error!("`this` is not a string: {}", realm.heap.show_debug(*this)).into()
    })?;

    Ok(this_str
        .view()
        .get(index)
        .map(|ndx| Value::Number(*ndx as f64))
        .unwrap_or(Value::Undefined))
}

fn nf_String_prototype_valueOf(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    _args: &[Value],
) -> RunResult<Value> {
    realm
        .heap
        .as_primitive(*this)
        .filter(|x| matches!(x, Value::String(_)))
        .ok_or_else(|| {
            let message = "String.prototype.valueOf requires that 'this' be a String";
            let exc = make_exception(realm, "TypeError", message);
            RunError::Exception(exc)
        })
}

fn nf_String_prototype_toString(
    realm: &mut Realm,
    loader: &mut loader::Loader,
    this: &Value,
    _args: &[Value],
) -> RunResult<Value> {
    nf_String_prototype_valueOf(realm, loader, this, &[])
}

fn nf_Boolean(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    _this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    // TODO Provide correct implementation for `new Boolean(...)` in addition to
    // `Boolean(...)`
    let arg = args.first().copied().unwrap_or(Value::Undefined);
    let bool_val = to_boolean(arg, &realm.heap);
    Ok(Value::Bool(bool_val))
}

fn nf_Boolean_prototype_valueOf(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    _args: &[Value],
) -> RunResult<Value> {
    realm
        .heap
        .as_primitive(*this)
        .filter(|x| matches!(x, Value::Bool(_)))
        .ok_or_else(|| {
            let message = "Boolean.prototype.valueOf requires that 'this' be a Boolean";
            let exc = make_exception(realm, "TypeError", message);
            RunError::Exception(exc)
        })
}

fn nf_Boolean_prototype_toString(
    realm: &mut Realm,
    loader: &mut loader::Loader,
    this: &Value,
    _args: &[Value],
) -> RunResult<Value> {
    nf_Boolean_prototype_valueOf(realm, loader, this, &[])
}

fn nf_Function(
    _realm: &mut Realm,
    _loader: &mut loader::Loader,
    _this: &Value,
    _: &[Value],
) -> RunResult<Value> {
    todo!("builtin Function()")
}

/// See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/Object#return_value
fn nf_Object(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    _this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    let arg = args.first().copied().unwrap_or(Value::Undefined);
    // unlike `Object.prototype.valueOf`, conversion failures fall back to an empty object
    Ok(to_object(arg, &mut realm.heap)
        .unwrap_or_else(|| Value::Object(realm.heap.new_ordinary_object())))
}

fn nf_Object_prototype_valueOf(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    _args: &[Value],
) -> RunResult<Value> {
    // This implementation of valueOf only 'converts to object'.

    // In non-strict mode, 'this substitution' [1] kicks in before the call
    // to this function. In most cases, `this` will have already been boxed
    // (converted to object) by the time we get here.

    to_object_or_throw(*this, realm)

    // More interesting conversions happen in other prototypes (e.g. Number.prototype,
    // String.prototype, etc.)
}

fn nf_Object_prototype_toString(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    _args: &[Value],
) -> RunResult<Value> {
    let particle = match this {
        Value::Number(_) => "Number",
        Value::Bool(_) => "Bool",
        Value::Object(_) => "Object",
        Value::String(_) => "String",
        Value::Null => "Null",
        Value::Undefined => "Undefined",
        Value::Symbol(_) => "Symbol",
    };
    let jss = JSString::new_from_str(&format!("[object {particle}]"));
    let sid = realm.heap.new_string(jss);
    Ok(Value::String(sid))
}

fn nf_cash_print(
    realm: &mut Realm,
    loader: &mut loader::Loader,
    _this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    for arg in args {
        let val = to_string_or_throw(*arg, realm, loader)?;
        let str = realm.heap.as_str(val).unwrap().to_string();
        println!("{}", str);
    }
    Ok(Value::Undefined)
}

fn nf_Function_bind(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    args: &[Value],
) -> RunResult<Value> {
    let orig_closure = realm
        .heap
        .as_closure(*this)
        .ok_or_else(|| error!("not a function"))?;

    let mut closure: Closure = (**orig_closure).clone();
    closure.forced_this = Some(args.first().copied().unwrap_or(Value::Undefined));

    let new_obj_id = realm.heap.new_function(Rc::new(closure));
    Ok(Value::Object(new_obj_id))
}

fn nf_Function_prototype_toString(
    realm: &mut Realm,
    _loader: &mut loader::Loader,
    this: &Value,
    _args: &[Value],
) -> RunResult<Value> {
    // TODO Return the function's code as string? This could work by using the loader's API...
    if realm.heap.as_closure(*this).is_some() {
        let jss = JSString::new_from_str("<Function>");
        let sid = realm.heap.new_string(jss);
        Ok(Value::String(sid))
    } else {
        let message = "Function.prototype.valueOf requires that 'this' be a Function";
        let exc = make_exception(realm, "TypeError", message);
        Err(RunError::Exception(exc))
    }
}
