// fsm_pnr_wrap.v — P&R-ONLY wrapper around the exported register-file FSM.
//
// WHY: the exported `lgn_fsm` (fsm.v) has a 1024-bit `q` output port => 1035
// top-level port bits, which exceeds the ~210 user IOBs of ANY xc7a35t package,
// so `lgn_fsm` cannot be placed as-is (the I/O-bound caveat in report.md §6/§7.4).
// This wrapper makes the FSM CORE placeable for a timing-only P&R run WITHOUT
// touching fsm.v: it instantiates lgn_fsm unchanged and reduces the 1024 state
// bits to a 32-bit REGISTERED signature (each output bit = parity of a 32-bit
// slice => a shallow 2-LUT6 reduction). All 1024 q bits stay live, so no FSM
// register is optimised away, and the FSM's own q -> next-state -> q paths are
// bit-identical to fsm.v. The reduction is a SEPARATE registered path; the
// nextpnr critical-path report names the offending net so we can confirm whether
// the reported Fmax is set by the FSM core (g<id> nets) or by the wrapper's XOR
// tree (yr). Not a deployable design; a timing probe for the FSM core in
// isolation, complementing the deployable lgn_top run.
module fsm_pnr_wrap (
    input  wire        clk,
    input  wire        rst,
    input  wire [8:0]  x,
    output wire [31:0] y
);
    wire [1023:0] q;
    lgn_fsm u_fsm (.clk(clk), .rst(rst), .x(x), .q(q));

    reg  [31:0] yr;
    integer i;
    always @(posedge clk) begin
        for (i = 0; i < 32; i = i + 1)
            yr[i] <= ^q[i*32 +: 32];   // parity of a contiguous 32-bit slice
    end
    assign y = yr;
endmodule
