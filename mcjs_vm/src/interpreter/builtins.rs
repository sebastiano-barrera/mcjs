#![allow(non_snake_case)]

use super::{literal_to_value, value_to_string, Closure, Interpreter, JSClosure, Value};

use crate::bytecode;
use crate::common::Result;
use crate::error;
use crate::heap::{self, IndexOrKey, Object};

use std::collections::HashMap;

pub(super) fn init_builtins(heap: &mut heap::Heap) -> heap::ObjectId {
    let mut global = HashMap::new();

    {
        let Array_push = heap.new_function(Closure::Native(nf_Array_push));
        let Array_pop = heap.new_function(Closure::Native(nf_Array_pop));
        let array_proto_oid = heap.array_proto();
        {
            let mut array_proto = heap.get(array_proto_oid).unwrap().borrow_mut();
            array_proto.set_own_element_or_property("push".into(), Value::Object(Array_push));
            array_proto.set_own_element_or_property("pop".into(), Value::Object(Array_pop));
        }

        let Array_isArray = heap.new_function(Closure::Native(nf_Array_isArray));
        let mut array_ctor_props = HashMap::new();
        array_ctor_props.insert("isArray".to_string(), Value::Object(Array_isArray));
        array_ctor_props.insert("prototype".to_string(), Value::Object(array_proto_oid));
        let array_ctor = heap.new_ordinary_object(array_ctor_props);
        heap.init_function(array_ctor, Closure::Native(nf_Array));

        global.insert("Array".to_string(), Value::Object(array_ctor));
    }

    let RegExp = heap.new_function(Closure::Native(nf_RegExp));
    global.insert("RegExp".to_string(), Value::Object(RegExp));

    let mut number_cons_props = HashMap::new();
    number_cons_props.insert("prototype".to_string(), Value::Object(heap.number_proto()));
    {
        let Number_prototype_toString =
            heap.new_function(Closure::Native(nf_Number_prototype_toString));
        let oid = heap.number_proto();
        let mut number_proto = heap.get(oid).unwrap().borrow_mut();
        number_proto.set_own_element_or_property(
            "toString".into(),
            Value::Object(Number_prototype_toString),
        )
    }
    let Number = heap.new_ordinary_object(number_cons_props);
    heap.init_function(Number, Closure::Native(nf_Number));
    global.insert("Number".to_string(), Value::Object(Number));

    let String = heap.new_function(Closure::Native(nf_String));
    global.insert("String".to_string(), Value::Object(String));

    let Boolean = heap.new_function(Closure::Native(nf_Boolean));
    global.insert("Boolean".to_string(), Value::Object(Boolean));

    let mut func_cons_props = HashMap::new();
    func_cons_props.insert("prototype".to_string(), Value::Object(heap.func_proto()));
    let Function = heap.new_ordinary_object(func_cons_props);
    heap.init_function(Function, Closure::Native(nf_Function));
    global.insert("Function".to_string(), Value::Object(Function));

    let cash_print = heap.new_function(Closure::Native(nf_cash_print));
    global.insert("$print".to_string(), Value::Object(cash_print));

    let func_bind = heap.new_function(Closure::Native(nf_Function_bind));
    {
        let mut func_proto = heap.get(heap.func_proto()).unwrap().borrow_mut();
        func_proto.set_own_element_or_property("bind".into(), Value::Object(func_bind));
    }

    {
        let oid = heap.new_ordinary_object(HashMap::new());
        global.insert("ReferenceError".to_string(), Value::Object(oid));
    }

    // builtins.insert("Boolean".into(), NativeFnId::BooleanNew as u32);
    // builtins.insert("Object".into(), NativeFnId::ObjectNew as u32);
    // builtins.insert("parseInt".into(), NativeFnId::ParseInt as u32);
    // builtins.insert("SyntaxError".into(), NativeFnId::SyntaxErrorNew as u32);
    // builtins.insert("TypeError".into(), NativeFnId::TypeErrorNew as u32);
    // builtins.insert("Math_floor".into(), NativeFnId::MathFloor as u32);

    // TODO(big feat) pls impl all Node.js API, ok? thxbye

    heap.new_ordinary_object(global)
}

fn nf_Array_isArray(intrp: &mut Interpreter, _this: &Value, args: &[Value]) -> Result<Value> {
    let value = if let Some(Value::Object(oid)) = args.get(0) {
        let obj = intrp
            .realm
            .heap
            .get(*oid)
            .ok_or_else(|| error!("no such object!"))?;
        obj.borrow().array_elements().is_some()
    } else {
        false
    };

    Ok(Value::Bool(value))
}

fn nf_Array_push(intrp: &mut Interpreter, this: &Value, args: &[Value]) -> Result<Value> {
    // TODO Proper error handling, instead of these unwrap
    let oid = this.expect_obj().unwrap();
    let mut arr = intrp.realm.heap.get(oid).unwrap().borrow_mut();
    let value = *args.get(0).unwrap();
    arr.array_push(value);
    Ok(Value::Undefined)
}

fn nf_Array_pop(_intrp: &mut Interpreter, _this: &Value, _args: &[Value]) -> Result<Value> {
    todo!("nf_Array_pop")
}

fn nf_RegExp(intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    // TODO
    let oid = intrp.realm.heap.new_ordinary_object(HashMap::new());
    Ok(Value::Object(oid))
}

fn nf_Array(intrp: &mut Interpreter, this: &Value, args: &[Value]) -> Result<Value> {
    let this = this
        .expect_obj()
        .expect("compiler bug: new Array(...): non-object this");
    // args directly go as values
    intrp.realm.heap.init_array(this, args.to_vec());
    Ok(Value::Undefined)
}

fn nf_Number(_intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    Ok(Value::Number(0.0))
}

fn nf_Number_prototype_toString(
    intrp: &mut Interpreter,
    this: &Value,
    _: &[Value],
) -> Result<Value> {
    let num_value = match this {
        Value::Number(num_value) => num_value,
        _ => return Err(error!("Not a number value!")),
    };

    let num_str = num_value.to_string();
    let oid = intrp.realm.heap.new_string(num_str);
    Ok(Value::Object(oid))
}

fn nf_String(intrp: &mut Interpreter, _this: &Value, args: &[Value]) -> Result<Value> {
    let value = args.get(0).copied();
    let heap = &intrp.realm.heap;

    let value_str = value
        .map(|v| value_to_string(v, heap))
        .unwrap_or_else(String::new);

    Ok(literal_to_value(
        bytecode::Literal::String(value_str),
        &mut intrp.realm.heap,
    ))
}

fn nf_Boolean(_intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    Ok(Value::Bool(false))
}

fn nf_Function(_intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    todo!("not yet implemented!")
}

fn nf_cash_print(intrp: &mut Interpreter, _this: &Value, args: &[Value]) -> Result<Value> {
    for arg in args {
        if let Value::Object(obj_id) = arg {
            let obj = intrp.realm.heap.get(*obj_id).unwrap().borrow();
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

fn nf_Function_bind(intrp: &mut Interpreter, this: &Value, args: &[Value]) -> Result<Value> {
    let js_closure = {
        let obj_id = this.expect_obj()?;
        let obj = intrp
            .realm
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

    let new_obj_id = intrp.realm.heap.new_function(new_closure);
    Ok(Value::Object(new_obj_id))
}
