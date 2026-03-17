`timescale 1ns / 1ps
//==============================================================================
// Module:      vga_controller
// File:        vga_controller.v
// Description: VGA sprite controller with button-controlled effects
//
// Features:
//   - 640x480 @ 60Hz VGA timing
//   - 8x16 pixel characters (80x30 display)
//   - Four sprites with independent position/color control
//   - Button effects: color toggle, size toggle, motion modes
//   - Intersection detection (blue when sprites overlap)
//   - All FFs have reset values
//
// Sprite Positions (centered):
//   Sprite 0: (288, 68)  - row 68
//   Sprite 1: (288, 181) - row 181
//   Sprite 2: (288, 298) - row 298
//   Sprite 3: (288, 411) - row 411
//
// Motion Boundaries:
//   Left:   20, Right:  619 (since 620+width would exceed)
//   Top:    20, Bottom: 459 (since 460+height would exceed)
//
// Velocities:
//   Ping-pong: 128 pixels/sec = 2.133 pixels/frame @ 60fps = 546 (2.133 * 256)
//   Wrap:      120 pixels/sec = 2.0 pixels/frame @ 60fps = 512 (2.0 * 256)
//==============================================================================

module vga_controller (
    input  wire        clk_vga,
    input  wire        resetn,

    // Button inputs (raw - debounced internally)
    input  wire        BTNC,
    input  wire        BTNU,
    input  wire        BTND,
    input  wire        BTNL,
    input  wire        BTNR,

    // Display data - 128-bit bus containing 4 × 32-bit values
    input  wire [127:0] display_words,
    input  wire         display_valid,

    // VGA output
    output reg  [3:0]  RED,
    output reg  [3:0]  GRN,
    output reg  [3:0]  BLU,
    output wire        HSYNC,
    output wire        VSYNC
);

    // =========================================================================
    // VGA Timing Constants (640x480 @ 60Hz)
    // =========================================================================
    localparam H_VIS   = 640;
    localparam H_FP    = 16;
    localparam H_SYNC  = 96;
    localparam H_BP    = 48;
    localparam H_TOTAL = 800;

    localparam V_VIS   = 480;
    localparam V_FP    = 10;
    localparam V_SYNC  = 2;
    localparam V_BP    = 33;
    localparam V_TOTAL = 525;

    // =========================================================================
    // Sprite Parameters (Q9.8 signed fixed-point: bits[17:8]=pixel, [7:0]=frac)
    // =========================================================================
    // Default positions - centered at column 320, rows at 68, 181, 298, 411
    localparam signed [17:0] DEF_X  = 18'sd73728;   // 288 << 8 = 73728
    localparam signed [17:0] DEF_Y0 = 18'sd17408;   // 68 << 8 = 17408
    localparam signed [17:0] DEF_Y1 = 18'sd46336;   // 181 << 8 = 46336
    localparam signed [17:0] DEF_Y2 = 18'sd76288;   // 298 << 8 = 76288
    localparam signed [17:0] DEF_Y3 = 18'sd105216;  // 411 << 8 = 105216

    // Motion boundaries (in integer pixels)
    localparam signed [9:0] BOUND_LEFT  = 10'sd20;
    localparam signed [9:0] BOUND_RIGHT = 10'sd619;  // 640 - 20 - width (will subtract sprite width)
    localparam signed [9:0] BOUND_TOP   = 10'sd20;
    localparam signed [9:0] BOUND_BOTTOM = 10'sd459; // 480 - 20 - height

    // Velocities (fixed-point Q9.8)
    localparam signed [17:0] VEL_PINGPONG = 18'sd546;  // 128 pixels/sec = 2.133 px/frame
    localparam signed [17:0] VEL_WRAP     = 18'sd512;  // 120 pixels/sec = 2.0 px/frame
    localparam signed [17:0] ZERO_VEL     = 18'sd0;

    // Sprite dimensions
    localparam [9:0] NORM_W = 10'd64;   // 8 chars × 8 pixels
    localparam [9:0] NORM_H = 10'd16;   // 16 pixels tall
    localparam [9:0] DBL_W  = 10'd128;  // 8 chars × 16 pixels
    localparam [9:0] DBL_H  = 10'd32;   // 32 pixels tall

    // Motion mode encoding
    localparam [1:0] MODE_STATIC   = 2'd0;
    localparam [1:0] MODE_PINGPONG = 2'd1;
    localparam [1:0] MODE_WRAP     = 2'd2;

    // =========================================================================
    // Raster Counters
    // =========================================================================
    reg [9:0] hc_ff;
    reg [9:0] vc_ff;

    always @(posedge clk_vga) begin
        if (!resetn) begin
            hc_ff <= 10'd0;
            vc_ff <= 10'd0;
        end else begin
            if (hc_ff == H_TOTAL - 1) begin
                hc_ff <= 10'd0;
                if (vc_ff == V_TOTAL - 1)
                    vc_ff <= 10'd0;
                else
                    vc_ff <= vc_ff + 10'd1;
            end else begin
                hc_ff <= hc_ff + 10'd1;
            end
        end
    end

    // Sync signals (pure combinational)
    assign HSYNC = ~( (hc_ff >= (H_VIS + H_FP)) && (hc_ff < (H_VIS + H_FP + H_SYNC)) );
    assign VSYNC = ~( (vc_ff >= (V_VIS + V_FP)) && (vc_ff < (V_VIS + V_FP + V_SYNC)) );

    wire visible = (hc_ff < H_VIS) && (vc_ff < V_VIS);

    // Frame tick on falling edge of VSYNC
    reg vsync_d_ff;
    always @(posedge clk_vga) begin
        if (!resetn)
            vsync_d_ff <= 1'b1;
        else
            vsync_d_ff <= VSYNC;
    end
    wire frame_tick = vsync_d_ff & ~VSYNC;

    // =========================================================================
    // 2FF CDC: display_words (clk_cpu) -> clk_vga
    // =========================================================================
    (* ASYNC_REG = "TRUE" *) reg [127:0] dw_sync1_ff;
    (* ASYNC_REG = "TRUE" *) reg [127:0] dw_sync2_ff;

    always @(posedge clk_vga) begin
        if (!resetn) begin
            dw_sync1_ff <= 128'h0;
            dw_sync2_ff <= 128'h0;
        end else begin
            dw_sync1_ff <= display_words;
            dw_sync2_ff <= dw_sync1_ff;
        end
    end

    // Each sprite gets its own 32-bit slice
    wire [31:0] sprite0_val = dw_sync2_ff[127:96];
    wire [31:0] sprite1_val = dw_sync2_ff[95:64];
    wire [31:0] sprite2_val = dw_sync2_ff[63:32];
    wire [31:0] sprite3_val = dw_sync2_ff[31:0];

    // =========================================================================
    // Debounce all 5 buttons
    // =========================================================================
    wire pulse_c, pulse_u, pulse_d, pulse_l, pulse_r;

    debounce u_dbc (.clk(clk_vga), .resetn(resetn), .btn_in(BTNC), .pulse_out(pulse_c));
    debounce u_dbu (.clk(clk_vga), .resetn(resetn), .btn_in(BTNU), .pulse_out(pulse_u));
    debounce u_dbd (.clk(clk_vga), .resetn(resetn), .btn_in(BTND), .pulse_out(pulse_d));
    debounce u_dbl (.clk(clk_vga), .resetn(resetn), .btn_in(BTNL), .pulse_out(pulse_l));
    debounce u_dbr (.clk(clk_vga), .resetn(resetn), .btn_in(BTNR), .pulse_out(pulse_r));

    // =========================================================================
    // Control State Registers
    // =========================================================================
    reg        active_ff;        // 0 = black screen, 1 = display active
    reg        first_u_ff;        // Has BTNU been pressed at least once?
    reg        red13_ff;          // 1 = sprites 0&2 are red, 0 = white
    reg        grn24_ff;          // 1 = sprites 1&3 are green, 0 = white
    reg        dbl_ff;            // 1 = double size, 0 = normal size
    reg [1:0]  mode_ff;           // 0=static, 1=pingpong, 2=wrap
    reg [1:0]  btnr_cnt_ff;       // BTNR press counter (0-3)

    // =========================================================================
    // Sprite Position Registers (Q9.8 signed fixed-point)
    // =========================================================================
    reg signed [17:0] px0_ff, py0_ff;
    reg signed [17:0] px1_ff, py1_ff;
    reg signed [17:0] px2_ff, py2_ff;
    reg signed [17:0] px3_ff, py3_ff;

    reg signed [17:0] vx0_ff, vy0_ff;
    reg signed [17:0] vx1_ff, vy1_ff;
    reg signed [17:0] vx2_ff, vy2_ff;
    reg signed [17:0] vx3_ff, vy3_ff;

    // Integer pixel positions (truncated fractional part)
    wire [9:0] ix0 = px0_ff[17:8];
    wire [9:0] iy0 = py0_ff[17:8];
    wire [9:0] ix1 = px1_ff[17:8];
    wire [9:0] iy1 = py1_ff[17:8];
    wire [9:0] ix2 = px2_ff[17:8];
    wire [9:0] iy2 = py2_ff[17:8];
    wire [9:0] ix3 = px3_ff[17:8];
    wire [9:0] iy3 = py3_ff[17:8];

    // Current sprite dimensions
    wire [9:0] sp_w = dbl_ff ? DBL_W : NORM_W;
    wire [9:0] sp_h = dbl_ff ? DBL_H : NORM_H;

    // =========================================================================
    // Control Logic
    // =========================================================================
    always @(posedge clk_vga) begin
        if (!resetn || pulse_c) begin
            // Reset to known state
            active_ff   <= 1'b0;
            first_u_ff  <= 1'b0;
            red13_ff    <= 1'b0;
            grn24_ff    <= 1'b0;
            dbl_ff      <= 1'b0;
            mode_ff     <= MODE_STATIC;
            btnr_cnt_ff <= 2'd0;
            
            // Reset positions to default
            px0_ff <= DEF_X;  py0_ff <= DEF_Y0;
            px1_ff <= DEF_X;  py1_ff <= DEF_Y1;
            px2_ff <= DEF_X;  py2_ff <= DEF_Y2;
            px3_ff <= DEF_X;  py3_ff <= DEF_Y3;
            
            // Reset velocities to zero
            vx0_ff <= ZERO_VEL;  vy0_ff <= ZERO_VEL;
            vx1_ff <= ZERO_VEL;  vy1_ff <= ZERO_VEL;
            vx2_ff <= ZERO_VEL;  vy2_ff <= ZERO_VEL;
            vx3_ff <= ZERO_VEL;  vy3_ff <= ZERO_VEL;
            
        end else begin
            //------------------------------------------------------------------------
            // BTNU: First press activates display, subsequent toggles red for 0&2
            //------------------------------------------------------------------------
            if (pulse_u) begin
                if (!first_u_ff) begin
                    active_ff  <= 1'b1;      // First press: activate display
                    first_u_ff <= 1'b1;
                end else begin
                    red13_ff <= ~red13_ff;    // Subsequent: toggle red
                end
            end

            //------------------------------------------------------------------------
            // BTND: Toggle green for sprites 1&3
            //------------------------------------------------------------------------
            if (pulse_d && active_ff) begin
                grn24_ff <= ~grn24_ff;
            end

            //------------------------------------------------------------------------
            // BTNL: Toggle double size
            //------------------------------------------------------------------------
            if (pulse_l && active_ff) begin
                dbl_ff <= ~dbl_ff;
            end

            //------------------------------------------------------------------------
            // BTNR: Cycle through motion modes
            //   Press 1: Ping-pong mode
            //   Press 2: Static mode (back to default)
            //   Press 3: Wrap mode
            //   Press 4: Static mode
            //   Press 5: Ping-pong again, etc.
            //------------------------------------------------------------------------
            if (pulse_r && active_ff) begin
                btnr_cnt_ff <= btnr_cnt_ff + 2'd1;
                
                case (btnr_cnt_ff)
                    2'd0: begin  // First press -> Ping-pong
                        mode_ff <= MODE_PINGPONG;
                        // Set initial velocities for ping-pong mode
                        vx0_ff <= -VEL_PINGPONG;  vy0_ff <= -VEL_PINGPONG;  // up-left
                        vx1_ff <=  VEL_PINGPONG;  vy1_ff <= -VEL_PINGPONG;  // up-right
                        vx2_ff <= -VEL_PINGPONG;  vy2_ff <=  VEL_PINGPONG;  // down-left
                        vx3_ff <=  VEL_PINGPONG;  vy3_ff <=  VEL_PINGPONG;  // down-right
                    end
                    
                    2'd1: begin  // Second press -> Static
                        mode_ff <= MODE_STATIC;
                        // Return to default positions
                        px0_ff <= DEF_X;  py0_ff <= DEF_Y0;
                        px1_ff <= DEF_X;  py1_ff <= DEF_Y1;
                        px2_ff <= DEF_X;  py2_ff <= DEF_Y2;
                        px3_ff <= DEF_X;  py3_ff <= DEF_Y3;
                        // Zero velocities
                        vx0_ff <= ZERO_VEL;  vy0_ff <= ZERO_VEL;
                        vx1_ff <= ZERO_VEL;  vy1_ff <= ZERO_VEL;
                        vx2_ff <= ZERO_VEL;  vy2_ff <= ZERO_VEL;
                        vx3_ff <= ZERO_VEL;  vy3_ff <= ZERO_VEL;
                    end
                    
                    2'd2: begin  // Third press -> Wrap
                        mode_ff <= MODE_WRAP;
                        // Set initial velocities for wrap mode
                        vx0_ff <= ZERO_VEL;      vy0_ff <=  VEL_WRAP;   // down
                        vx1_ff <= ZERO_VEL;      vy1_ff <= -VEL_WRAP;   // up
                        vx2_ff <= ZERO_VEL;      vy2_ff <=  VEL_WRAP;   // down
                        vx3_ff <= ZERO_VEL;      vy3_ff <= -VEL_WRAP;   // up
                        // Keep X positions at default
                        px0_ff <= DEF_X;  px1_ff <= DEF_X;
                        px2_ff <= DEF_X;  px3_ff <= DEF_X;
                    end
                    
                    2'd3: begin  // Fourth press -> Static
                        mode_ff <= MODE_STATIC;
                        // Return to default positions
                        px0_ff <= DEF_X;  py0_ff <= DEF_Y0;
                        px1_ff <= DEF_X;  py1_ff <= DEF_Y1;
                        px2_ff <= DEF_X;  py2_ff <= DEF_Y2;
                        px3_ff <= DEF_X;  py3_ff <= DEF_Y3;
                        // Zero velocities
                        vx0_ff <= ZERO_VEL;  vy0_ff <= ZERO_VEL;
                        vx1_ff <= ZERO_VEL;  vy1_ff <= ZERO_VEL;
                        vx2_ff <= ZERO_VEL;  vy2_ff <= ZERO_VEL;
                        vx3_ff <= ZERO_VEL;  vy3_ff <= ZERO_VEL;
                    end
                endcase
            end

            //------------------------------------------------------------------------
            // Update positions every frame (when active and in motion modes)
            //------------------------------------------------------------------------
            if (frame_tick && active_ff) begin
                case (mode_ff)
                    MODE_PINGPONG: begin
                        // Update all positions
                        px0_ff <= px0_ff + vx0_ff;
                        py0_ff <= py0_ff + vy0_ff;
                        px1_ff <= px1_ff + vx1_ff;
                        py1_ff <= py1_ff + vy1_ff;
                        px2_ff <= px2_ff + vx2_ff;
                        py2_ff <= py2_ff + vy2_ff;
                        px3_ff <= px3_ff + vx3_ff;
                        py3_ff <= py3_ff + vy3_ff;

                        // Sprite 0 boundary checking
                        if (ix0 < BOUND_LEFT) begin
                            px0_ff <= {BOUND_LEFT, 8'h00};
                            vx0_ff <= -vx0_ff;
                        end else if ((ix0 + sp_w - 1) > BOUND_RIGHT) begin
                            px0_ff <= {(BOUND_RIGHT - sp_w + 1), 8'h00};
                            vx0_ff <= -vx0_ff;
                        end
                        
                        if (iy0 < BOUND_TOP) begin
                            py0_ff <= {BOUND_TOP, 8'h00};
                            vy0_ff <= -vy0_ff;
                        end else if ((iy0 + sp_h - 1) > BOUND_BOTTOM) begin
                            py0_ff <= {(BOUND_BOTTOM - sp_h + 1), 8'h00};
                            vy0_ff <= -vy0_ff;
                        end

                        // Sprite 1 boundary checking
                        if (ix1 < BOUND_LEFT) begin
                            px1_ff <= {BOUND_LEFT, 8'h00};
                            vx1_ff <= -vx1_ff;
                        end else if ((ix1 + sp_w - 1) > BOUND_RIGHT) begin
                            px1_ff <= {(BOUND_RIGHT - sp_w + 1), 8'h00};
                            vx1_ff <= -vx1_ff;
                        end
                        
                        if (iy1 < BOUND_TOP) begin
                            py1_ff <= {BOUND_TOP, 8'h00};
                            vy1_ff <= -vy1_ff;
                        end else if ((iy1 + sp_h - 1) > BOUND_BOTTOM) begin
                            py1_ff <= {(BOUND_BOTTOM - sp_h + 1), 8'h00};
                            vy1_ff <= -vy1_ff;
                        end

                        // Sprite 2 boundary checking
                        if (ix2 < BOUND_LEFT) begin
                            px2_ff <= {BOUND_LEFT, 8'h00};
                            vx2_ff <= -vx2_ff;
                        end else if ((ix2 + sp_w - 1) > BOUND_RIGHT) begin
                            px2_ff <= {(BOUND_RIGHT - sp_w + 1), 8'h00};
                            vx2_ff <= -vx2_ff;
                        end
                        
                        if (iy2 < BOUND_TOP) begin
                            py2_ff <= {BOUND_TOP, 8'h00};
                            vy2_ff <= -vy2_ff;
                        end else if ((iy2 + sp_h - 1) > BOUND_BOTTOM) begin
                            py2_ff <= {(BOUND_BOTTOM - sp_h + 1), 8'h00};
                            vy2_ff <= -vy2_ff;
                        end

                        // Sprite 3 boundary checking
                        if (ix3 < BOUND_LEFT) begin
                            px3_ff <= {BOUND_LEFT, 8'h00};
                            vx3_ff <= -vx3_ff;
                        end else if ((ix3 + sp_w - 1) > BOUND_RIGHT) begin
                            px3_ff <= {(BOUND_RIGHT - sp_w + 1), 8'h00};
                            vx3_ff <= -vx3_ff;
                        end
                        
                        if (iy3 < BOUND_TOP) begin
                            py3_ff <= {BOUND_TOP, 8'h00};
                            vy3_ff <= -vy3_ff;
                        end else if ((iy3 + sp_h - 1) > BOUND_BOTTOM) begin
                            py3_ff <= {(BOUND_BOTTOM - sp_h + 1), 8'h00};
                            vy3_ff <= -vy3_ff;
                        end
                    end

                    MODE_WRAP: begin
                        // Update Y positions only
                        py0_ff <= py0_ff + vy0_ff;
                        py1_ff <= py1_ff + vy1_ff;
                        py2_ff <= py2_ff + vy2_ff;
                        py3_ff <= py3_ff + vy3_ff;

                        // Wrap checking
                        // Sprite 0 (down) - wrap from bottom to top
                        if ((iy0 + sp_h - 1) >= BOUND_BOTTOM) begin
                            py0_ff <= {BOUND_TOP, 8'h00};
                        end
                        
                        // Sprite 1 (up) - wrap from top to bottom
                        if (iy1 <= BOUND_TOP) begin
                            py1_ff <= {(BOUND_BOTTOM - sp_h + 1), 8'h00};
                        end
                        
                        // Sprite 2 (down) - wrap from bottom to top
                        if ((iy2 + sp_h - 1) >= BOUND_BOTTOM) begin
                            py2_ff <= {BOUND_TOP, 8'h00};
                        end
                        
                        // Sprite 3 (up) - wrap from top to bottom
                        if (iy3 <= BOUND_TOP) begin
                            py3_ff <= {(BOUND_BOTTOM - sp_h + 1), 8'h00};
                        end
                    end

                    default: begin
                        // MODE_STATIC: No position updates
                    end
                endcase
            end
        end
    end

    // =========================================================================
    // Font ROM: 8x16 pixels per character (0-9, A-F)
    // =========================================================================
    function [7:0] font_row;
        input [3:0] ch;
        input [3:0] row;
        begin
            case (ch)
                4'h0: begin
                    case (row)
                        4'd0:  font_row = 8'b00111100;
                        4'd1:  font_row = 8'b01000010;
                        4'd2:  font_row = 8'b01000110;
                        4'd3:  font_row = 8'b01001010;
                        4'd4:  font_row = 8'b01010010;
                        4'd5:  font_row = 8'b01100010;
                        4'd6:  font_row = 8'b01000010;
                        4'd7:  font_row = 8'b01000010;
                        4'd8:  font_row = 8'b01000010;
                        4'd9:  font_row = 8'b01000010;
                        4'd10: font_row = 8'b01000010;
                        4'd11: font_row = 8'b01000010;
                        4'd12: font_row = 8'b01000010;
                        4'd13: font_row = 8'b01000010;
                        4'd14: font_row = 8'b00111100;
                        default: font_row = 8'h00;
                    endcase
                end
                4'h1: begin
                    case (row)
                        4'd0:  font_row = 8'b00011000;
                        4'd1:  font_row = 8'b00111000;
                        4'd2:  font_row = 8'b00011000;
                        4'd3:  font_row = 8'b00011000;
                        4'd4:  font_row = 8'b00011000;
                        4'd5:  font_row = 8'b00011000;
                        4'd6:  font_row = 8'b00011000;
                        4'd7:  font_row = 8'b00011000;
                        4'd8:  font_row = 8'b00011000;
                        4'd9:  font_row = 8'b00011000;
                        4'd10: font_row = 8'b00011000;
                        4'd11: font_row = 8'b00011000;
                        4'd12: font_row = 8'b00011000;
                        4'd13: font_row = 8'b00011000;
                        4'd14: font_row = 8'b00111100;
                        default: font_row = 8'h00;
                    endcase
                end
                4'h2: begin
                    case (row)
                        4'd0:  font_row = 8'b00111100;
                        4'd1:  font_row = 8'b01000010;
                        4'd2:  font_row = 8'b00000010;
                        4'd3:  font_row = 8'b00000010;
                        4'd4:  font_row = 8'b00000100;
                        4'd5:  font_row = 8'b00001000;
                        4'd6:  font_row = 8'b00010000;
                        4'd7:  font_row = 8'b00100000;
                        4'd8:  font_row = 8'b01000000;
                        4'd9:  font_row = 8'b01000000;
                        4'd10: font_row = 8'b01000000;
                        4'd11: font_row = 8'b01000000;
                        4'd12: font_row = 8'b01000000;
                        4'd13: font_row = 8'b01000000;
                        4'd14: font_row = 8'b01111110;
                        default: font_row = 8'h00;
                    endcase
                end
                4'h3: begin
                    case (row)
                        4'd0:  font_row = 8'b00111100;
                        4'd1:  font_row = 8'b01000010;
                        4'd2:  font_row = 8'b00000010;
                        4'd3:  font_row = 8'b00000010;
                        4'd4:  font_row = 8'b00001100;
                        4'd5:  font_row = 8'b00000010;
                        4'd6:  font_row = 8'b00000010;
                        4'd7:  font_row = 8'b00000010;
                        4'd8:  font_row = 8'b00000010;
                        4'd9:  font_row = 8'b00000010;
                        4'd10: font_row = 8'b00000010;
                        4'd11: font_row = 8'b00000010;
                        4'd12: font_row = 8'b00000010;
                        4'd13: font_row = 8'b01000010;
                        4'd14: font_row = 8'b00111100;
                        default: font_row = 8'h00;
                    endcase
                end
                4'h4: begin
                    case (row)
                        4'd0:  font_row = 8'b00000100;
                        4'd1:  font_row = 8'b00001100;
                        4'd2:  font_row = 8'b00010100;
                        4'd3:  font_row = 8'b00100100;
                        4'd4:  font_row = 8'b01000100;
                        4'd5:  font_row = 8'b01000100;
                        4'd6:  font_row = 8'b01000100;
                        4'd7:  font_row = 8'b01111110;
                        4'd8:  font_row = 8'b00000100;
                        4'd9:  font_row = 8'b00000100;
                        4'd10: font_row = 8'b00000100;
                        4'd11: font_row = 8'b00000100;
                        4'd12: font_row = 8'b00000100;
                        4'd13: font_row = 8'b00000100;
                        4'd14: font_row = 8'b00000100;
                        default: font_row = 8'h00;
                    endcase
                end
                4'h5: begin
                    case (row)
                        4'd0:  font_row = 8'b01111110;
                        4'd1:  font_row = 8'b01000000;
                        4'd2:  font_row = 8'b01000000;
                        4'd3:  font_row = 8'b01000000;
                        4'd4:  font_row = 8'b01111100;
                        4'd5:  font_row = 8'b00000010;
                        4'd6:  font_row = 8'b00000010;
                        4'd7:  font_row = 8'b00000010;
                        4'd8:  font_row = 8'b00000010;
                        4'd9:  font_row = 8'b00000010;
                        4'd10: font_row = 8'b00000010;
                        4'd11: font_row = 8'b00000010;
                        4'd12: font_row = 8'b00000010;
                        4'd13: font_row = 8'b01000010;
                        4'd14: font_row = 8'b00111100;
                        default: font_row = 8'h00;
                    endcase
                end
                4'h6: begin
                    case (row)
                        4'd0:  font_row = 8'b00111100;
                        4'd1:  font_row = 8'b01000010;
                        4'd2:  font_row = 8'b01000000;
                        4'd3:  font_row = 8'b01000000;
                        4'd4:  font_row = 8'b01111100;
                        4'd5:  font_row = 8'b01000010;
                        4'd6:  font_row = 8'b01000010;
                        4'd7:  font_row = 8'b01000010;
                        4'd8:  font_row = 8'b01000010;
                        4'd9:  font_row = 8'b01000010;
                        4'd10: font_row = 8'b01000010;
                        4'd11: font_row = 8'b01000010;
                        4'd12: font_row = 8'b01000010;
                        4'd13: font_row = 8'b01000010;
                        4'd14: font_row = 8'b00111100;
                        default: font_row = 8'h00;
                    endcase
                end
                4'h7: begin
                    case (row)
                        4'd0:  font_row = 8'b01111110;
                        4'd1:  font_row = 8'b00000010;
                        4'd2:  font_row = 8'b00000010;
                        4'd3:  font_row = 8'b00000100;
                        4'd4:  font_row = 8'b00000100;
                        4'd5:  font_row = 8'b00001000;
                        4'd6:  font_row = 8'b00001000;
                        4'd7:  font_row = 8'b00010000;
                        4'd8:  font_row = 8'b00010000;
                        4'd9:  font_row = 8'b00100000;
                        4'd10: font_row = 8'b00100000;
                        4'd11: font_row = 8'b01000000;
                        4'd12: font_row = 8'b01000000;
                        4'd13: font_row = 8'b01000000;
                        4'd14: font_row = 8'b01000000;
                        default: font_row = 8'h00;
                    endcase
                end
                4'h8: begin
                    case (row)
                        4'd0:  font_row = 8'b00111100;
                        4'd1:  font_row = 8'b01000010;
                        4'd2:  font_row = 8'b01000010;
                        4'd3:  font_row = 8'b01000010;
                        4'd4:  font_row = 8'b00111100;
                        4'd5:  font_row = 8'b01000010;
                        4'd6:  font_row = 8'b01000010;
                        4'd7:  font_row = 8'b01000010;
                        4'd8:  font_row = 8'b01000010;
                        4'd9:  font_row = 8'b01000010;
                        4'd10: font_row = 8'b01000010;
                        4'd11: font_row = 8'b01000010;
                        4'd12: font_row = 8'b01000010;
                        4'd13: font_row = 8'b01000010;
                        4'd14: font_row = 8'b00111100;
                        default: font_row = 8'h00;
                    endcase
                end
                4'h9: begin
                    case (row)
                        4'd0:  font_row = 8'b00111100;
                        4'd1:  font_row = 8'b01000010;
                        4'd2:  font_row = 8'b01000010;
                        4'd3:  font_row = 8'b01000010;
                        4'd4:  font_row = 8'b00111110;
                        4'd5:  font_row = 8'b00000010;
                        4'd6:  font_row = 8'b00000010;
                        4'd7:  font_row = 8'b00000010;
                        4'd8:  font_row = 8'b00000010;
                        4'd9:  font_row = 8'b00000010;
                        4'd10: font_row = 8'b00000010;
                        4'd11: font_row = 8'b00000010;
                        4'd12: font_row = 8'b00000010;
                        4'd13: font_row = 8'b01000010;
                        4'd14: font_row = 8'b00111100;
                        default: font_row = 8'h00;
                    endcase
                end
                
                4'hA: begin
                    case (row)
                        4'd0:  font_row = 8'b00011000;
                        4'd1:  font_row = 8'b00100100;
                        4'd2:  font_row = 8'b01000010;
                        4'd3:  font_row = 8'b01000010;
                        4'd4:  font_row = 8'b01000010;
                        4'd5:  font_row = 8'b01111110;
                        4'd6:  font_row = 8'b01000010;
                        4'd7:  font_row = 8'b01000010;
                        4'd8:  font_row = 8'b01000010;
                        4'd9:  font_row = 8'b01000010;
                        4'd10: font_row = 8'b01000010;
                        4'd11: font_row = 8'b01000010;
                        4'd12: font_row = 8'b01000010;
                        4'd13: font_row = 8'b01000010;
                        4'd14: font_row = 8'b01000010;
                        default: font_row = 8'h00;
                    endcase
                end
                
                4'hB: begin
                    case (row)
                        4'd0:  font_row = 8'b01111100;
                        4'd1:  font_row = 8'b01000010;
                        4'd2:  font_row = 8'b01000010;
                        4'd3:  font_row = 8'b01000010;
                        4'd4:  font_row = 8'b01111100;
                        4'd5:  font_row = 8'b01000010;
                        4'd6:  font_row = 8'b01000010;
                        4'd7:  font_row = 8'b01000010;
                        4'd8:  font_row = 8'b01000010;
                        4'd9:  font_row = 8'b01000010;
                        4'd10: font_row = 8'b01000010;
                        4'd11: font_row = 8'b01000010;
                        4'd12: font_row = 8'b01000010;
                        4'd13: font_row = 8'b01000010;
                        4'd14: font_row = 8'b01111100;
                        default: font_row = 8'h00;
                    endcase
                end
                
                4'hC: begin
                    case (row)
                        4'd0:  font_row = 8'b00111100;
                        4'd1:  font_row = 8'b01000010;
                        4'd2:  font_row = 8'b01000000;
                        4'd3:  font_row = 8'b01000000;
                        4'd4:  font_row = 8'b01000000;
                        4'd5:  font_row = 8'b01000000;
                        4'd6:  font_row = 8'b01000000;
                        4'd7:  font_row = 8'b01000000;
                        4'd8:  font_row = 8'b01000000;
                        4'd9:  font_row = 8'b01000000;
                        4'd10: font_row = 8'b01000000;
                        4'd11: font_row = 8'b01000000;
                        4'd12: font_row = 8'b01000000;
                        4'd13: font_row = 8'b01000010;
                        4'd14: font_row = 8'b00111100;
                        default: font_row = 8'h00;
                    endcase
                end
                
                4'hD: begin
                    case (row)
                        4'd0:  font_row = 8'b01111100;
                        4'd1:  font_row = 8'b01000010;
                        4'd2:  font_row = 8'b01000010;
                        4'd3:  font_row = 8'b01000010;
                        4'd4:  font_row = 8'b01000010;
                        4'd5:  font_row = 8'b01000010;
                        4'd6:  font_row = 8'b01000010;
                        4'd7:  font_row = 8'b01000010;
                        4'd8:  font_row = 8'b01000010;
                        4'd9:  font_row = 8'b01000010;
                        4'd10: font_row = 8'b01000010;
                        4'd11: font_row = 8'b01000010;
                        4'd12: font_row = 8'b01000010;
                        4'd13: font_row = 8'b01000010;
                        4'd14: font_row = 8'b01111100;
                        default: font_row = 8'h00;
                    endcase
                end
                
                4'hE: begin
                    case (row)
                        4'd0:  font_row = 8'b01111110;
                        4'd1:  font_row = 8'b01000000;
                        4'd2:  font_row = 8'b01000000;
                        4'd3:  font_row = 8'b01000000;
                        4'd4:  font_row = 8'b01111100;
                        4'd5:  font_row = 8'b01000000;
                        4'd6:  font_row = 8'b01000000;
                        4'd7:  font_row = 8'b01000000;
                        4'd8:  font_row = 8'b01000000;
                        4'd9:  font_row = 8'b01000000;
                        4'd10: font_row = 8'b01000000;
                        4'd11: font_row = 8'b01000000;
                        4'd12: font_row = 8'b01000000;
                        4'd13: font_row = 8'b01000000;
                        4'd14: font_row = 8'b01111110;
                        default: font_row = 8'h00;
                    endcase
                end
                
                4'hF: begin
                    case (row)
                        4'd0:  font_row = 8'b01111110;
                        4'd1:  font_row = 8'b01000000;
                        4'd2:  font_row = 8'b01000000;
                        4'd3:  font_row = 8'b01000000;
                        4'd4:  font_row = 8'b01111100;
                        4'd5:  font_row = 8'b01000000;
                        4'd6:  font_row = 8'b01000000;
                        4'd7:  font_row = 8'b01000000;
                        4'd8:  font_row = 8'b01000000;
                        4'd9:  font_row = 8'b01000000;
                        4'd10: font_row = 8'b01000000;
                        4'd11: font_row = 8'b01000000;
                        4'd12: font_row = 8'b01000000;
                        4'd13: font_row = 8'b01000000;
                        4'd14: font_row = 8'b01000000;
                        default: font_row = 8'h00;
                    endcase
                end
                
                default: font_row = 8'h00;
            endcase
        end
    endfunction

    // =========================================================================
    // Sprite Hit Detection Function
    // =========================================================================
    function sprite_hit;
        input [9:0]  px, py;      // Current pixel position
        input [9:0]  sx, sy;      // Sprite top-left position
        input [31:0] val;         // 32-bit sprite value (8 hex digits)
        input        dbl;         // Double size flag
        
        reg [9:0]  offx, offy;
        reg [2:0]  cidx;          // Character index (0-7)
        reg [2:0]  col;           // Column within character (0-7)
        reg [3:0]  row;           // Row within character (0-15)
        reg [3:0]  nib;           // Nibble value (0-F)
        reg [7:0]  bits;          // Font row bits
        begin
            sprite_hit = 1'b0;
            
            if (dbl) begin
                // Double size: 16x32 per character (2x scaling)
                if (px >= sx && px < sx + 10'd128 &&
                    py >= sy && py < sy + 10'd32) begin
                    
                    offx = px - sx;
                    offy = py - sy;
                    cidx = offx[6:4];        // Which character (0-7)
                    col  = offx[3:1];         // Which column (0-7) - divide by 2
                    row  = offy[4:1];         // Which row (0-15) - divide by 2
                    
                    // Get the nibble for this character
                    case (cidx)
                        3'd0: nib = val[31:28];
                        3'd1: nib = val[27:24];
                        3'd2: nib = val[23:20];
                        3'd3: nib = val[19:16];
                        3'd4: nib = val[15:12];
                        3'd5: nib = val[11:8];
                        3'd6: nib = val[7:4];
                        3'd7: nib = val[3:0];
                        default: nib = 4'h0;
                    endcase
                    
                    bits = font_row(nib, row);
                    sprite_hit = bits[3'd7 - col];
                end
            end else begin
                // Normal size: 8x16 per character
                if (px >= sx && px < sx + 10'd64 &&
                    py >= sy && py < sy + 10'd16) begin
                    
                    offx = px - sx;
                    offy = py - sy;
                    cidx = offx[5:3];        // Which character (0-7)
                    col  = offx[2:0];         // Which column (0-7)
                    row  = offy[3:0];         // Which row (0-15)
                    
                    // Get the nibble for this character
                    case (cidx)
                        3'd0: nib = val[31:28];
                        3'd1: nib = val[27:24];
                        3'd2: nib = val[23:20];
                        3'd3: nib = val[19:16];
                        3'd4: nib = val[15:12];
                        3'd5: nib = val[11:8];
                        3'd6: nib = val[7:4];
                        3'd7: nib = val[3:0];
                        default: nib = 4'h0;
                    endcase
                    
                    bits = font_row(nib, row);
                    sprite_hit = bits[3'd7 - col];
                end
            end
        end
    endfunction

    // =========================================================================
    // Per-Pixel Hit Signals
    // =========================================================================
    wire h0 = sprite_hit(hc_ff, vc_ff, ix0, iy0, sprite0_val, dbl_ff);
    wire h1 = sprite_hit(hc_ff, vc_ff, ix1, iy1, sprite1_val, dbl_ff);
    wire h2 = sprite_hit(hc_ff, vc_ff, ix2, iy2, sprite2_val, dbl_ff);
    wire h3 = sprite_hit(hc_ff, vc_ff, ix3, iy3, sprite3_val, dbl_ff);

    // Count how many sprites are hitting this pixel
    wire [2:0] hit_count = {2'b00, h0} + {2'b00, h1} + {2'b00, h2} + {2'b00, h3};
    wire intersect = (hit_count >= 3'd2);  // Two or more sprites overlap

    // =========================================================================
    // Pixel Color Output
    // =========================================================================
    always @(posedge clk_vga) begin
        if (!resetn) begin
            RED <= 4'h0;
            GRN <= 4'h0;
            BLU <= 4'h0;
            
        end else if (!visible || !active_ff) begin
            // Outside visible area or display inactive: BLACK
            RED <= 4'h0;
            GRN <= 4'h0;
            BLU <= 4'h0;
            
        end else if (intersect) begin
            // Intersection: BLUE
            RED <= 4'h0;
            GRN <= 4'h0;
            BLU <= 4'hF;
            
        end else if (h0 || h2) begin
            // Sprites 0 and 2: WHITE normally, RED when toggled
            RED <= 4'hF;
            GRN <= red13_ff ? 4'h0 : 4'hF;
            BLU <= red13_ff ? 4'h0 : 4'hF;
            
        end else if (h1 || h3) begin
            // Sprites 1 and 3: WHITE normally, GREEN when toggled
            RED <= grn24_ff ? 4'h0 : 4'hF;
            GRN <= 4'hF;
            BLU <= grn24_ff ? 4'h0 : 4'hF;
            
        end else begin
            // Background: BLACK
            RED <= 4'h0;
            GRN <= 4'h0;
            BLU <= 4'h0;
        end
    end

endmodule