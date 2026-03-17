`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 01/05/2026 06:13:57 PM
// Design Name: 
// Module Name: sd_sync
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module sd_sync(
    input wire         clk_cpu,
    input wire         resetn,
    input wire  [31:0] sd_data,
    input wire         sd_lineflag,
    output wire        cpu_lineflag_pulse,
    output wire [31:0] cpu_data
);
    
    reg  [2:0]  cpu_lineflag_sync;
    reg  [31:0] cpu_data_s1, cpu_data_s2;
    
    // Synchronize data (stable data)
    always @(posedge clk_cpu or negedge resetn) begin
        if (!resetn) begin
            cpu_data_s1 <= 32'b0;
            cpu_data_s2 <= 32'b0;
        end else begin
            cpu_data_s1 <= sd_data;
            cpu_data_s2 <= cpu_data_s1;
        end
    end
    assign cpu_data = cpu_data_s2;
    
    // Synchronize lineflag toggle and detect changes
    always @(posedge clk_cpu or negedge resetn) begin
        if (!resetn) begin
            cpu_lineflag_sync <= 3'b0;
        end else begin
            cpu_lineflag_sync <= {cpu_lineflag_sync[1:0], sd_lineflag};
        end
    end
    assign cpu_lineflag_pulse = cpu_lineflag_sync[2];

endmodule
