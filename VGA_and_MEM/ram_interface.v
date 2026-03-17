
module ram_interface(
    input wire clk_mem,
    input wire clk_cpu,
    input resetn,
    input lineflag,
    input [27:0] mem_addr_in,
    input [63:0] mem_d_to_ram_in,
    output [63:0] mem_d_from_ram_out,
    output ramflag,
    input rflag,
    input wflag,
    output readfini,
    output writefini,
    output wire [15:0] led,
    
    // DDR2 pins
    inout[15:0] ddr2_dq,
    inout[1:0] ddr2_dqs_n,
    inout[1:0] ddr2_dqs_p,
    output[12:0] ddr2_addr,
    output[2:0] ddr2_ba,
    output ddr2_ras_n,
    output ddr2_cas_n,
    output ddr2_we_n,
    output ddr2_ck_p,
    output ddr2_ck_n,
    output ddr2_cke,
    output ddr2_cs_n,
    output[1:0] ddr2_dm,
    output ddr2_odt
);

    // Direct pass-through
    wire writingstart;
    wire readingstart;
    wire rflagf;
    wire wflagf;
    
    assign writingstart = wflag;
    assign readingstart = rflag;
    assign readfini = rflagf;
    assign writefini = wflagf;
    
    mig_example_top u_mig_example_top (
        .clk_mem(clk_mem),
        .clk_cpu(clk_cpu),
        .CPU_RESETN(resetn),
        .lineflag(lineflag),
        .mem_addr_in(mem_addr_in),
        .mem_d_to_ram_in(mem_d_to_ram_in),
        .mem_d_from_ram(mem_d_from_ram_out),
        .ramflag(ramflag),
        .rflag(rflag),
        .wflag(wflag),
        .writingstart(writingstart),
        .readingstart(readingstart),
        .rflagf(rflagf),
        .wflagf(wflagf),
        .led(led),
        .ddr2_dq(ddr2_dq),
        .ddr2_dqs_n(ddr2_dqs_n),
        .ddr2_dqs_p(ddr2_dqs_p),
        .ddr2_addr(ddr2_addr),
        .ddr2_ba(ddr2_ba),
        .ddr2_ras_n(ddr2_ras_n),
        .ddr2_cas_n(ddr2_cas_n),
        .ddr2_we_n(ddr2_we_n),
        .ddr2_ck_p(ddr2_ck_p),
        .ddr2_ck_n(ddr2_ck_n),
        .ddr2_cke(ddr2_cke),
        .ddr2_cs_n(ddr2_cs_n),
        .ddr2_dm(ddr2_dm),
        .ddr2_odt(ddr2_odt)
    );

endmodule