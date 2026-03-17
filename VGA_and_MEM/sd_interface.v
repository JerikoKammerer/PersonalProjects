module sd_interface(
    input wire          clk_sd,
    input  wire         resetn,
    output wire         sdcard_pwr_n,
    output wire         sdclk,
    inout               sdcmd,
    input  wire         sddat0,
    output wire         sddat1, sddat2, sddat3,
    output wire [15:0]  led,
    output reg  [31:0]  data,
    input wire          sd_run,
    output reg          sd_finish,
    output wire         lineflag
);
    
    // Connect to fpga_top
    wire [7:0] outbyte;
    wire [15:0] ledout;
    wire [2:0] filesystem_state;
    wire outen;
    
    assign led = ledout;
    
    fpga_top u_fpga_top (
        .clk (clk_sd),
        .resetn (sd_run),           // Use sd_run directly as reset
        .sdcard_pwr_n (sdcard_pwr_n),
        .sdclk (sdclk),
        .sdcmd (sdcmd),
        .sddat0 (sddat0),
        .sddat1 (sddat1),
        .sddat2 (sddat2),
        .sddat3 (sddat3),
        .ledout (ledout),
        .outen (outen),
        .outbyte (outbyte),
        .filesystem_state (filesystem_state)
    );
    
    // State machine
    localparam [3:0] FIRST   = 4'd0,
                     SECOND  = 4'd1,
                     THIRD   = 4'd2,
                     FOURTH  = 4'd3,
                     FIFTH   = 4'd4,
                     SIXTH   = 4'd5,
                     SEVENTH = 4'd6,
                     EIGHTH  = 4'd7,
                     NINTH   = 4'd8;
    
    reg [3:0] sdstate;
    reg lineflag_toggle;
    assign lineflag = lineflag_toggle;
    
    // ASCII to hex/decimal conversion
    wire [3:0] ascii2dec = outbyte - 8'd48;
    wire [3:0] ascii2hex = outbyte - 8'd87;
    
    // Finish signal
    always @(posedge clk_sd or negedge resetn) begin
        if (!resetn)
            sd_finish <= 1'b0;
        else
            sd_finish <= (filesystem_state == 3'd6);
    end
    
    assign sd_run = (!resetn) ? 1'b0 : 1'b1;
    
    // ========================================================================
    // STATE MACHINE (processes data in bytes)
    // ========================================================================
    always @(posedge clk_sd or negedge resetn) begin
        if (!resetn) begin
            sdstate <= FIRST;
            data <= 32'h0;
            lineflag_toggle <= 1'b0;
        end else begin
            // Only process when outen is HIGH (indicates new byte available)
            if (outen) begin
                case (sdstate)
                    FIRST: begin
                        // First hex digit (MSB of data)
                        if (outbyte > 8'd60)  // 'a'-'f' or 'A'-'F'
                            data[31:28] <= ascii2hex;
                        else                  // '0'-'9'
                            data[31:28] <= ascii2dec;
                        sdstate <= SECOND;
                    end
                    
                    SECOND: begin
                        if (outbyte > 8'd60)
                            data[27:24] <= ascii2hex;
                        else
                            data[27:24] <= ascii2dec;
                        sdstate <= THIRD;
                    end
                    
                    THIRD: begin
                        if (outbyte > 8'd60)
                            data[23:20] <= ascii2hex;
                        else
                            data[23:20] <= ascii2dec;
                        sdstate <= FOURTH;
                    end
                    
                    FOURTH: begin
                        if (outbyte > 8'd60)
                            data[19:16] <= ascii2hex;
                        else
                            data[19:16] <= ascii2dec;
                        sdstate <= FIFTH;
                    end
                    
                    FIFTH: begin
                        if (outbyte > 8'd60)
                            data[15:12] <= ascii2hex;
                        else
                            data[15:12] <= ascii2dec;
                        sdstate <= SIXTH;
                    end
                    
                    SIXTH: begin
                        if (outbyte > 8'd60)
                            data[11:8] <= ascii2hex;
                        else
                            data[11:8] <= ascii2dec;
                        sdstate <= SEVENTH;
                    end
                    
                    SEVENTH: begin
                        if (outbyte > 8'd60)
                            data[7:4] <= ascii2hex;
                        else
                            data[7:4] <= ascii2dec;
                        sdstate <= EIGHTH;
                    end
                    
                    EIGHTH: begin
                        // Last hex digit (LSB of data)
                        if (outbyte > 8'd60)
                            data[3:0] <= ascii2hex;
                        else
                            data[3:0] <= ascii2dec;
                        
                        // Toggle lineflag to indicate complete data
                        lineflag_toggle <= ~lineflag_toggle;
                        sdstate <= NINTH;
                    end
                    
                    NINTH: begin
                        // This state receives the newline character ('\n' = 0x0A)
                        sdstate <= FIRST;
                    end
                    
                    default: begin
                        sdstate <= FIRST;
                    end
                endcase
            end
            // If outen is LOW, maintain current state (wait for next byte)
        end
    end
    
endmodule
