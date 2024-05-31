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
            let mut array_proto_obj = heap.get_mut(array_proto).unwrap();
            array_proto_obj.set_own("push".into(), heap::Property::non_enumerable(Array_push));
            array_proto_obj.set_own("pop".into(), heap::Property::non_enumerable(Array_pop));
        }

        let Array_isArray = Value::Object(heap.new_function(Closure::Native(nf_Array_isArray)));
        let array_ctor = Value::Object(heap.new_function(Closure::Native(nf_Array)));
        {
            let mut array_ctor_obj = heap.get_mut(array_ctor).unwrap();
            array_ctor_obj.set_own(
                "isArray".into(),
                heap::Property::non_enumerable(Array_isArray),
            );
            array_ctor_obj.set_own(
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

        let mut Number_obj = heap.get_mut(Number).unwrap();
        Number_obj.set_own(
            "prototype".into(),
            heap::Property::non_enumerable(Value::Object(heap.number_proto())),
        );

        let mut number_proto_obj = heap.get_mut(Value::Object(heap.number_proto())).unwrap();
        number_proto_obj.set_own("toString".into(), heap::Property::non_enumerable(toString))
    }

    let String = Value::Object(heap.new_function(Closure::Native(nf_String)));

    let Boolean = Value::Object(heap.new_function(Closure::Native(nf_Boolean)));

    let Function = Value::Object(heap.new_function(Closure::Native(nf_Function)));
    {
        let mut Function_obj = heap.get_mut(Function).unwrap();
        Function_obj.set_own(
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
        let mut func_proto = heap.get_mut(Value::Object(heap.func_proto())).unwrap();
        func_proto.set_own("bind".into(), heap::Property::non_enumerable(func_bind));
    }

    let Error = Value::Object(heap.new_function(Closure::Native(nf_do_nothing)));
    {
        let Error_toString = Value::Object(heap.new_function(Closure::Native(nf_Error_toString)));
        let mut Error_obj = heap.get_mut(Error).unwrap();
        Error_obj.set_own(
            heap::IndexOrKey::Key("toString"),
            heap::Property::enumerable(Error_toString),
        );
    }
    let ReferenceError = make_exception_cons(heap, Error);
    let TypeError = make_exception_cons(heap, Error);
    let SyntaxError = make_exception_cons(heap, Error);

    // TODO(big feat) pls impl all Node.js API, ok? thxbye

    let global = Value::Object(heap.new_ordinary_object());
    let mut global_obj = heap.get_mut(global).unwrap();
    global_obj.set_own("Object".into(), heap::Property::enumerable(Object));
    global_obj.set_own("Array".into(), heap::Property::enumerable(Array));
    global_obj.set_own("RegExp".into(), heap::Property::enumerable(RegExp));
    global_obj.set_own("Number".into(), heap::Property::enumerable(Number));
    global_obj.set_own("String".into(), heap::Property::enumerable(String));
    global_obj.set_own("Boolean".into(), heap::Property::enumerable(Boolean));
    global_obj.set_own("Function".into(), heap::Property::enumerable(Function));
    global_obj.set_own("$print".into(), heap::Property::enumerable(cash_print));
    global_obj.set_own("Error".into(), heap::Property::enumerable(Error));
    global_obj.set_own(
        "ReferenceError".into(),
        heap::Property::enumerable(ReferenceError),
    );
    global_obj.set_own("TypeError".into(), heap::Property::enumerable(TypeError));
    global_obj.set_own(
        "SyntaxError".into(),
        heap::Property::enumerable(SyntaxError),
    );

    global
}

fn make_exception_cons(heap: &mut heap::Heap, prototype: Value) -> Value {
    let cons = Value::Object(heap.new_function(Closure::Native(nf_set_message)));
    let mut cons_obj = heap.get_mut(cons).unwrap();
    cons_obj.set_own(
        heap::IndexOrKey::Key("prototype"),
        heap::Property::non_enumerable(prototype),
    );
    cons
}

fn nf_set_message(realm: &mut Realm, this: &Value, args: &[Value]) -> Result<Value> {
    let message = *args.get(0).unwrap_or(&Value::Undefined);
    let mut obj = realm.heap.get_mut(*this).expect("bug: not an object!?");
    obj.set_own("message".into(), heap::Property::enumerable(message));
    Ok(Value::Undefined)
}

fn nf_do_nothing(_realm: &mut Realm, _this: &Value, _args: &[Value]) -> Result<Value> {
    Ok(Value::Undefined)
}

fn nf_Array_isArray(realm: &mut Realm, _this: &Value, args: &[Value]) -> Result<Value> {
    let value = if let Some(obj) = args.first() {
        let obj = realm
            .heap
            .get(*obj)
            .ok_or_else(|| error!("no such object!"))?;
        obj.array_elements().is_some()
    } else {
        false
    };

    Ok(Value::Bool(value))
}

fn nf_Array_push(realm: &mut Realm, this: &Value, args: &[Value]) -> Result<Value> {
    // TODO Proper error handling, instead of these unwrap
    let mut arr = realm.heap.get_mut(*this).unwrap();
    let value = *args.first().unwrap();
    arr.array_push(value);
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
            let obj = realm.heap.get_mut(Value::Object(*oid));
            obj.unwrap().set_proto(Some(value_str));
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
    let obj = realm
        .heap
        .get(*this)
        .ok_or_else(|| error!("no such object"))?;
    let closure = obj.as_closure().ok_or_else(|| error!("not a function"))?;
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

    drop(obj);

    let new_obj_id = realm.heap.new_function(Closure::JS(Rc::new(new_closure)));
    Ok(Value::Object(new_obj_id))
}

fn nf_Error_toString(realm: &mut Realm, this: &Value, _args: &[Value]) -> Result<Value> {
    let obj = realm.heap.get(*this).unwrap();

    let message = obj
        .get_chained(heap::IndexOrKey::Key("message"))
        .map(|prop| value_to_string(prop.value, &realm.heap))
        .unwrap_or(Ok(JSString::new_from_str("(no message)")))?;
    let name = obj
        .get_chained(heap::IndexOrKey::Key("name"))
        .map(|prop| value_to_string(prop.value, &realm.heap))
        .unwrap_or(Ok(JSString::new_from_str("(no message)")))?;

    let mut full_message = Vec::new();
    full_message.extend_from_slice(name.view());
    full_message.extend(": ".encode_utf16());
    full_message.extend_from_slice(message.view());

    let full_message = JSString::new(full_message);
    drop(obj);

    let oid = realm.heap.new_string(full_message);
    Ok(Value::Object(oid))
}
