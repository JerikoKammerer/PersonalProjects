`timescale 1ns / 1ps
//==============================================================================
// Module:      debounce
// File:        debounce.v
// Description: Single-button debouncer with configurable counter
//
// Features:
//   - Two-FF synchronizer for metastability protection
//   - Counter-based debounce with programmable max count
//   - Outputs single-cycle pulse on stable rising edge
//   - All FFs have reset values
//
// Parameters:
//   SIM_MODE: 0 = normal (20ms debounce), 1 = simulation (16 cycles)
//
// Inputs:
//   clk      - Clock (use clk_vga = 25.175 MHz)
//   resetn   - Active-low reset
//   btn_in   - Raw button input (active-high)
//
// Outputs:
//   pulse_out - Single-cycle pulse on stable rising edge
//
// Timing:
//   For 25.175 MHz, 20ms = 503,500 cycles ~= 2^19 = 524,288
//==============================================================================

module debounce #(
    parameter SIM_MODE = 0
) (
    input  wire clk,
    input  wire resetn,
    input  wire btn_in,
    output reg  pulse_out
);

    // =========================================================================
    // Constants
    // =========================================================================
    // 20ms at 25.175 MHz = 503,500 cycles, round to 2^19 = 524,288
    localparam [19:0] CNT_MAX = (SIM_MODE) ? 20'd16 : 20'd524287;

    // =========================================================================
    // Two-FF Synchronizer (Metastability Protection)
    // =========================================================================
    reg s1_ff;
    reg s2_ff;
    
    always @(posedge clk) begin
        if (!resetn) begin
            s1_ff <= 1'b0;
            s2_ff <= 1'b0;
        end else begin
            s1_ff <= btn_in;
            s2_ff <= s1_ff;
        end
    end

    // =========================================================================
    // Debounce Counter and Stable State
    // =========================================================================
    reg [19:0] cnt_ff;
    reg        stable_ff;

    always @(posedge clk) begin
        if (!resetn) begin
            cnt_ff    <= 20'd0;
            stable_ff <= 1'b0;
            pulse_out <= 1'b0;
        end else begin
            // Default: deassert pulse
            pulse_out <= 1'b0;
            
            if (s2_ff == stable_ff) begin
                // Button matches stable state: reset counter
                cnt_ff <= 20'd0;
            end else begin
                // Button differs from stable state: count
                if (cnt_ff == CNT_MAX) begin
                    // Debounce time reached: update stable state
                    cnt_ff    <= 20'd0;
                    stable_ff <= s2_ff;
                    
                    // Generate pulse only on rising edge
                    if (s2_ff == 1'b1) begin
                        pulse_out <= 1'b1;
                    end
                end else begin
                    // Continue counting
                    cnt_ff <= cnt_ff + 20'd1;
                end
            end
        end
    end

endmodule
