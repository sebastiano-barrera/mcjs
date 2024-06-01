#![allow(non_snake_case)]

use std::rc::Rc;

use super::{show_value, value_to_string, Closure, JSClosure, Realm, Value};

use crate::common::Result;
use crate::error;
use crate::heap;
use crate::heap::JSString;

pub(super) fn init_builtins(heap: &mut heap::Heap) -> Value {
    let Array = {
        let Array_push = Value::Object(heap.new_function(Closure::Native(nf_Array_push)));
        let Array_pop = Value::Object(heap.new_function(Closure::Native(nf_Array_pop)));
        let array_proto = Value::Object(heap.array_proto());
        {
            heap.set_own(
                array_proto,
                "push".into(),
                heap::Property::non_enumerable(Array_push),
            );
            heap.set_own(
                array_proto,
                "pop".into(),
                heap::Property::non_enumerable(Array_pop),
            );
        }

        let Array_isArray = Value::Object(heap.new_function(Closure::Native(nf_Array_isArray)));
        let array_ctor = Value::Object(heap.new_function(Closure::Native(nf_Array)));
        {
            heap.set_own(
                array_ctor,
                "isArray".into(),
                heap::Property::non_enumerable(Array_isArray),
            );
            heap.set_own(
                array_ctor,
                "prototype".into(),
                heap::Property::non_enumerable(array_proto),
            );
        }

        array_ctor
    };

    let RegExp = Value::Object(heap.new_function(Closure::Native(nf_RegExp)));

    let Number = Value::Object(heap.new_function(Closure::Native(nf_Number)));
    {
        let toString =
            Value::Object(heap.new_function(Closure::Native(nf_Number_prototype_toString)));

        heap.set_own(
            Number,
            "prototype".into(),
            heap::Property::non_enumerable(Value::Object(heap.number_proto())),
        );

        heap.set_own(
            Value::Object(heap.number_proto()),
            "toString".into(),
            heap::Property::non_enumerable(toString),
        )
    }

    let String = Value::Object(heap.new_function(Closure::Native(nf_String)));

    let Boolean = Value::Object(heap.new_function(Closure::Native(nf_Boolean)));

    let Function = Value::Object(heap.new_function(Closure::Native(nf_Function)));
    {
        heap.set_own(
            Function,
            "prototype".into(),
            heap::Property::non_enumerable(Value::Object(heap.func_proto())),
        );
    }

    // Not completely correct. See the rules in
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/Object#return_value
    let Object = Value::Object(heap.new_function(Closure::Native(nf_do_nothing)));

    let cash_print = Value::Object(heap.new_function(Closure::Native(nf_cash_print)));

    let func_bind = Value::Object(heap.new_function(Closure::Native(nf_Function_bind)));
    {
        heap.set_own(
            Value::Object(heap.func_proto()),
            "bind".into(),
            heap::Property::non_enumerable(func_bind),
        );
    }

    let Error = Value::Object(heap.new_function(Closure::Native(nf_do_nothing)));
    {
        let Error_toString = Value::Object(heap.new_function(Closure::Native(nf_Error_toString)));
        heap.set_own(
            Error,
            heap::IndexOrKey::Key("toString"),
            heap::Property::enumerable(Error_toString),
        );
    }
    let ReferenceError = make_exception_cons(heap, Error);
    let TypeError = make_exception_cons(heap, Error);
    let SyntaxError = make_exception_cons(heap, Error);

    // TODO(big feat) pls impl all Node.js API, ok? thxbye

    let global = Value::Object(heap.new_ordinary_object());
    heap.set_own(global, "Object".into(), heap::Property::enumerable(Object));
    heap.set_own(global, "Array".into(), heap::Property::enumerable(Array));
    heap.set_own(global, "RegExp".into(), heap::Property::enumerable(RegExp));
    heap.set_own(global, "Number".into(), heap::Property::enumerable(Number));
    heap.set_own(global, "String".into(), heap::Property::enumerable(String));
    heap.set_own(
        global,
        "Boolean".into(),
        heap::Property::enumerable(Boolean),
    );
    heap.set_own(
        global,
        "Function".into(),
        heap::Property::enumerable(Function),
    );
    heap.set_own(
        global,
        "$print".into(),
        heap::Property::enumerable(cash_print),
    );
    heap.set_own(global, "Error".into(), heap::Property::enumerable(Error));
    heap.set_own(
        global,
        "ReferenceError".into(),
        heap::Property::enumerable(ReferenceError),
    );
    heap.set_own(
        global,
        "TypeError".into(),
        heap::Property::enumerable(TypeError),
    );
    heap.set_own(
        global,
        "SyntaxError".into(),
        heap::Property::enumerable(SyntaxError),
    );

    global
}

fn make_exception_cons(heap: &mut heap::Heap, prototype: Value) -> Value {
    let cons = Value::Object(heap.new_function(Closure::Native(nf_set_message)));
    heap.set_own(
        cons,
        heap::IndexOrKey::Key("prototype"),
        heap::Property::non_enumerable(prototype),
    );
    cons
}

fn nf_set_message(realm: &mut Realm, this: &Value, args: &[Value]) -> Result<Value> {
    let message = *args.get(0).unwrap_or(&Value::Undefined);
    realm
        .heap
        .set_own(*this, "message".into(), heap::Property::enumerable(message));
    Ok(Value::Undefined)
}

fn nf_do_nothing(_realm: &mut Realm, _this: &Value, _args: &[Value]) -> Result<Value> {
    Ok(Value::Undefined)
}

fn nf_Array_isArray(realm: &mut Realm, _this: &Value, args: &[Value]) -> Result<Value> {
    let value = if let Some(obj) = args.first() {
        realm.heap.array_elements(*obj).is_some()
    } else {
        false
    };

    Ok(Value::Bool(value))
}

fn nf_Array_push(realm: &mut Realm, this: &Value, args: &[Value]) -> Result<Value> {
    // TODO Proper error handling, instead of these unwrap
    let value = *args.first().unwrap();
    realm.heap.array_push(*this, value);
    Ok(Value::Undefined)
}

fn nf_Array_pop(_realm: &mut Realm, _this: &Value, _args: &[Value]) -> Result<Value> {
    todo!("nf_Array_pop")
}

fn nf_RegExp(realm: &mut Realm, _this: &Value, _: &[Value]) -> Result<Value> {
    // TODO
    let oid = realm.heap.new_ordinary_object();
    Ok(Value::Object(oid))
}

fn nf_Array(realm: &mut Realm, this: &Value, args: &[Value]) -> Result<Value> {
    let this = this
        .expect_obj()
        .expect("compiler bug: new Array(...): non-object this");
    // args directly go as values
    realm.heap.init_array(this, args.to_vec());
    Ok(Value::Undefined)
}

fn nf_Number(_realm: &mut Realm, _this: &Value, _: &[Value]) -> Result<Value> {
    Ok(Value::Number(0.0))
}

fn nf_Number_prototype_toString(realm: &mut Realm, this: &Value, _: &[Value]) -> Result<Value> {
    let num_value = match this {
        Value::Number(num_value) => num_value,
        _ => return Err(error!("Not a number value!")),
    };

    let num_str = JSString::new_from_str(&num_value.to_string());
    let oid = realm.heap.new_string(num_str);
    Ok(Value::Object(oid))
}

fn nf_String(realm: &mut Realm, this: &Value, args: &[Value]) -> Result<Value> {
    let value = args.first().copied();

    let value_str = match value {
        None => JSString::empty(),
        Some(v) => value_to_string(v, &realm.heap)?,
    };
    let value_str = realm.heap.new_string(value_str);

    match this {
        Value::Object(oid) => {
            // called as a constructor: string primitive as prototype of new ordinary object
            realm.heap.set_proto(Value::Object(*oid), Some(value_str));
            Ok(Value::Object(*oid))
        }
        Value::Undefined => {
            // called as function, not constructor -> return string primitive
            Ok(Value::Object(value_str))
        }
        _ => panic!("bug: 'this' not an object: {:?}", this),
    }
}

fn nf_Boolean(_realm: &mut Realm, _this: &Value, _: &[Value]) -> Result<Value> {
    Ok(Value::Bool(false))
}

fn nf_Function(_realm: &mut Realm, _this: &Value, _: &[Value]) -> Result<Value> {
    todo!("not yet implemented!")
}

fn nf_cash_print(realm: &mut Realm, _this: &Value, args: &[Value]) -> Result<Value> {
    let mut out = std::io::stdout().lock();
    for arg in args {
        show_value(&mut out, *arg, realm);
    }
    Ok(Value::Undefined)
}

fn nf_Function_bind(realm: &mut Realm, this: &Value, args: &[Value]) -> Result<Value> {
    let closure = realm
        .heap
        .as_closure(*this)
        .ok_or_else(|| error!("not a function"))?;
    let js_closure = match closure {
        Closure::Native(_) => return Err(error!("can't bind a native function (yet)")),
        Closure::JS(jsc) => jsc,
    };

    let forced_this = Some(args.first().copied().unwrap_or(Value::Undefined));
    let new_closure = JSClosure {
        forced_this,
        fnid: js_closure.fnid,
        upvalues: js_closure.upvalues.clone(),
        generator_snapshot: js_closure.generator_snapshot.clone(),
    };

    let new_obj_id = realm.heap.new_function(Closure::JS(Rc::new(new_closure)));
    Ok(Value::Object(new_obj_id))
}

fn nf_Error_toString(realm: &mut Realm, this: &Value, _args: &[Value]) -> Result<Value> {
    let message = realm.heap
        .get_chained(*this, heap::IndexOrKey::Key("message"))
        .map(|prop| value_to_string(prop.value, &realm.heap))
        .unwrap_or(Ok(JSString::new_from_str("(no message)")))?;
    let name = realm.heap
        .get_chained(*this, heap::IndexOrKey::Key("name"))
        .map(|prop| value_to_string(prop.value, &realm.heap))
        .unwrap_or(Ok(JSString::new_from_str("(no message)")))?;

    let mut full_message = Vec::new();
    full_message.extend_from_slice(name.view());
    full_message.extend(": ".encode_utf16());
    full_message.extend_from_slice(message.view());

    let full_message = JSString::new(full_message);
    let oid = realm.heap.new_string(full_message);
    Ok(Value::Object(oid))
}
