#![allow(non_snake_case)]

use super::{literal_to_value, value_to_string, Closure, JSClosure, Realm, Value};

use crate::bytecode;
use crate::common::Result;
use crate::error;
use crate::heap::{self, IndexOrKey, Object};

pub(super) fn init_builtins(heap: &mut heap::Heap) -> heap::ObjectId {
    let array_ctor = {
        let Array_push = heap.new_function(Closure::Native(nf_Array_push));
        let Array_pop = heap.new_function(Closure::Native(nf_Array_pop));
        let array_proto = heap.array_proto();
        {
            let mut array_proto_obj = heap.get(array_proto).unwrap().borrow_mut();
            array_proto_obj.set_own_element_or_property("push".into(), Value::Object(Array_push));
            array_proto_obj.set_own_element_or_property("pop".into(), Value::Object(Array_pop));
        }

        let Array_isArray = heap.new_function(Closure::Native(nf_Array_isArray));
        let array_ctor = heap.new_function(Closure::Native(nf_Array));
        {
            let mut array_ctor_obj = heap.get(array_ctor).unwrap().borrow_mut();
            array_ctor_obj
                .set_own_element_or_property("isArray".into(), Value::Object(Array_isArray));
            array_ctor_obj
                .set_own_element_or_property("prototype".into(), Value::Object(array_proto));
        }

        array_ctor
    };

    let RegExp = heap.new_function(Closure::Native(nf_RegExp));

    let Number = heap.new_function(Closure::Native(nf_Number));
    {
        let toString = heap.new_function(Closure::Native(nf_Number_prototype_toString));

        let mut Number_obj = heap.get(Number).unwrap().borrow_mut();
        Number_obj
            .set_own_element_or_property("prototype".into(), Value::Object(heap.number_proto()));

        let mut number_proto_obj = heap.get(heap.number_proto()).unwrap().borrow_mut();
        number_proto_obj.set_own_element_or_property("toString".into(), Value::Object(toString))
    }

    let String = heap.new_function(Closure::Native(nf_String));

    let Boolean = heap.new_function(Closure::Native(nf_Boolean));

    let Function = heap.new_function(Closure::Native(nf_Function));
    {
        let mut Function_obj = heap.get(Function).unwrap().borrow_mut();
        Function_obj
            .set_own_element_or_property("prototype".into(), Value::Object(heap.func_proto()));
    }

    let cash_print = heap.new_function(Closure::Native(nf_cash_print));

    let func_bind = heap.new_function(Closure::Native(nf_Function_bind));
    {
        let mut func_proto = heap.get(heap.func_proto()).unwrap().borrow_mut();
        func_proto.set_own_element_or_property("bind".into(), Value::Object(func_bind));
    }

    let ReferenceError = heap.new_ordinary_object();

    // TODO(big feat) pls impl all Node.js API, ok? thxbye

    let global = heap.new_ordinary_object();
    let mut global_obj = heap.get(global).unwrap().borrow_mut();
    global_obj.set_own_element_or_property("Array".into(), Value::Object(array_ctor));
    global_obj.set_own_element_or_property("RegExp".into(), Value::Object(RegExp));
    global_obj.set_own_element_or_property("Number".into(), Value::Object(Number));
    global_obj.set_own_element_or_property("String".into(), Value::Object(String));
    global_obj.set_own_element_or_property("Boolean".into(), Value::Object(Boolean));
    global_obj.set_own_element_or_property("Function".into(), Value::Object(Function));
    global_obj.set_own_element_or_property("$print".into(), Value::Object(cash_print));
    global_obj.set_own_element_or_property("ReferenceError".into(), Value::Object(ReferenceError));

    global
}

fn nf_Array_isArray(realm: &mut Realm, _this: &Value, args: &[Value]) -> Result<Value> {
    let value = if let Some(Value::Object(oid)) = args.get(0) {
        let obj = realm
            .heap
            .get(*oid)
            .ok_or_else(|| error!("no such object!"))?;
        obj.borrow().array_elements().is_some()
    } else {
        false
    };

    Ok(Value::Bool(value))
}

fn nf_Array_push(realm: &mut Realm, this: &Value, args: &[Value]) -> Result<Value> {
    // TODO Proper error handling, instead of these unwrap
    let oid = this.expect_obj().unwrap();
    let mut arr = realm.heap.get(oid).unwrap().borrow_mut();
    let value = *args.get(0).unwrap();
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

    let num_str = num_value.to_string();
    let oid = realm.heap.new_string(num_str);
    Ok(Value::Object(oid))
}

fn nf_String(realm: &mut Realm, _this: &Value, args: &[Value]) -> Result<Value> {
    let value = args.get(0).copied();
    let heap = &realm.heap;

    let value_str = value
        .map(|v| value_to_string(v, heap))
        .unwrap_or_else(String::new);

    Ok(literal_to_value(
        bytecode::Literal::String(value_str),
        &mut realm.heap,
    ))
}

fn nf_Boolean(_realm: &mut Realm, _this: &Value, _: &[Value]) -> Result<Value> {
    Ok(Value::Bool(false))
}

fn nf_Function(_realm: &mut Realm, _this: &Value, _: &[Value]) -> Result<Value> {
    todo!("not yet implemented!")
}

fn nf_cash_print(realm: &mut Realm, _this: &Value, args: &[Value]) -> Result<Value> {
    for arg in args {
        if let Value::Object(obj_id) = arg {
            let obj = realm.heap.get(*obj_id).unwrap().borrow();
            if let Some(s) = obj.as_str() {
                println!("  {:?}", s);
            } else {
                let props = obj.own_properties();
                println!("{:?} [{} properties]", obj_id, props.len());

                for prop in props {
                    let value = obj.get_own_element_or_property(IndexOrKey::Key(&prop));
                    println!("  - {:?} = {:?}", prop, value);
                }
            }
        } else {
            println!("{:?}", arg);
        }
    }
    Ok(Value::Undefined)
}

fn nf_Function_bind(realm: &mut Realm, this: &Value, args: &[Value]) -> Result<Value> {
    let js_closure = {
        let obj_id = this.expect_obj()?;
        let obj = realm
            .heap
            .get(obj_id)
            .ok_or_else(|| error!("no such object"))?
            .borrow();
        let closure = obj.as_closure().ok_or_else(|| error!("not a function"))?;
        match closure {
            Closure::Native(_) => return Err(error!("can't bind a native function (yet)")),
            Closure::JS(jsc) => jsc.clone(),
        }
    };

    let forced_this = Some(args.get(0).copied().unwrap_or(Value::Undefined));

    let new_closure = Closure::JS(JSClosure {
        forced_this,
        ..js_closure
    });

    let new_obj_id = realm.heap.new_function(new_closure);
    Ok(Value::Object(new_obj_id))
}
