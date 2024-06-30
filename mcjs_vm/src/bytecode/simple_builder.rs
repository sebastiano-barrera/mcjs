use crate::{
    bytecode::{ConstIndex, Function, FunctionBuilder, Instr, VReg},
    Literal, IID,
};

#[derive(Default)]
pub struct FnBuilder {
    consts: Vec<Literal>,
    instrs: Vec<Instr>,
    n_regs: u16,
}

impl FnBuilder {
    pub fn new() -> Self {
        FnBuilder::default()
    }

    pub fn add_const(&mut self, lit: Literal) -> ConstIndex {
        let ndx = self.consts.len().try_into().unwrap();
        self.consts.push(lit);
        ConstIndex(ndx)
    }

    pub fn set_instr(&mut self, iid: IID, instr: Instr) {
        self.instrs[iid.0 as usize] = instr;
    }

    pub fn peek_iid(&self) -> IID {
        IID(self.instrs.len().try_into().unwrap())
    }

    pub fn emit(&mut self, instr: Instr) {
        self.instrs.push(instr);
    }

    pub fn reserve_instr(&mut self) -> IID {
        let iid = self.peek_iid();
        self.emit(Instr::Nop);
        iid
    }

    pub fn gen_reg(&mut self) -> VReg {
        let reg = VReg(self.n_regs);
        self.n_regs += 1;
        reg
    }

    pub fn build(self) -> Function {
        FunctionBuilder {
            instrs: self.instrs.into_boxed_slice(),
            consts: self.consts.into_boxed_slice(),
            n_regs: self.n_regs,
            ident_history: Vec::new(),
            is_strict_mode: true,
            span: Default::default(),
        }
        .build()
    }
}
