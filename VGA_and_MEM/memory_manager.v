`timescale 1ns / 1ps
//==============================================================================
// Module:      memory_manager
// File:        memory_manager.v
// Description: Sits between ram_interface and vga_controller.
//              - Receives 4 x 32-bit values from the SD card one at a time
//                (each signalled by a lineflag toggle edge) and writes them
//                into DDR2 at addresses 0-3.
//              - After all 4 words are written, continuously reads them back
//                in a round-robin sequence, assembles the four 32-bit words
//                into a single 128-bit display_words bus, and asserts
//                display_valid so the VGA controller can latch the data.
//
// Clock domain notes:
//   - Everything here runs on clk_mem (the MIG user-interface clock).
//   - vga_controller already contains a 2-FF CDC on display_words, so
//     display_words and display_valid can be driven directly from clk_mem.
//
// Data layout in DDR2 (one 32-bit word per 64-bit location, zero-extended):
//   Address 0  -> sprite 0  (display_words[127:96])
//   Address 1  -> sprite 1  (display_words[ 95:64])
//   Address 2  -> sprite 2  (display_words[ 63:32])
//   Address 3  -> sprite 3  (display_words[ 31: 0])
//
// ram_interface protocol:
//   Write: assert wflag, put address on mem_addr_out and data on mem_data_out.
//          Wait for writefini to go HIGH, then deassert wflag.
//          Wait for writefini to go LOW before issuing the next transaction.
//   Read:  assert rflag, put address on mem_addr_out.
//          Wait for readfini to go HIGH, then deassert rflag and latch
//          mem_data_in.  Wait for readfini to go LOW before next transaction.
//==============================================================================

module memory_manager (
    input  wire         clk_mem,
    input  wire         resetn,

    // -------------------------------------------------------------------------
    // SD card interface (one 32-bit word per lineflag toggle edge)
    // -------------------------------------------------------------------------
    input  wire [31:0]  sd_data,
    input  wire         sd_lineflag,      // toggle-style flag from sd_interface

    // -------------------------------------------------------------------------
    // DDR2 / ram_interface
    // -------------------------------------------------------------------------
    output reg  [27:0]  mem_addr_out,
    output reg  [63:0]  mem_data_out,
    input  wire [63:0]  mem_data_in,
    output reg          wflag,
    output reg          rflag,
    input  wire         writefini,
    input  wire         readfini,

    // -------------------------------------------------------------------------
    // Output to vga_controller  (driven from clk_mem; VGA module re-syncs)
    // -------------------------------------------------------------------------
    output reg  [127:0] display_words,
    output reg          display_valid
);

    // =========================================================================
    // Detect rising/toggle edge of sd_lineflag (toggle-style from sd_interface)
    // =========================================================================
    reg  sd_lineflag_prev;
    wire sd_lineflag_pulse = sd_lineflag;

    always @(posedge clk_mem or negedge resetn) begin
        if (!resetn) sd_lineflag_prev <= 1'b0;
        else         sd_lineflag_prev <= sd_lineflag;
    end

    // =========================================================================
    // Latch incoming SD word and queue it for writing
    // =========================================================================
    reg [31:0] sd_word_lat;     // latched word to be written to DDR2
    reg        sd_pending;      // a word is waiting to be written

    // How many SD words have been received (0-4)
    reg [2:0]  sd_count;        // counts 0..4; saturates at 4
    reg        all_written;     // all 4 words have been committed to DDR2

    always @(posedge clk_mem or negedge resetn) begin
        if (!resetn) begin
            sd_word_lat <= 32'h0;
            sd_pending  <= 1'b0;
            sd_count    <= 3'd0;
            all_written <= 1'b0;
        end else begin
            // New word arrives from SD card (only accept the first 4)
            if (sd_lineflag_pulse && sd_count < 3'd4) begin
                sd_word_lat <= sd_data;
                sd_pending  <= 1'b1;
            end

            // Clear pending once the write FSM picks it up
            if (sd_pending && state == WRITE_REQ)
                sd_pending <= 1'b0;

            // Track how many words have been written to DDR2
            if (state == WRITE_WAIT && !writefini)
                if (sd_count < 3'd4)
                    sd_count <= sd_count + 3'd1;

            // All 4 words stored → enable continuous readback
            if (sd_count == 3'd4)
                all_written <= 1'b1;
        end
    end

    // =========================================================================
    // Assemble 128-bit frame from four 32-bit reads
    // =========================================================================
    reg [31:0] word_buf [0:3];  // staging buffer: word_buf[0..3]
    reg [1:0]  read_idx;        // which of the 4 words we are currently reading
    reg        frame_ready;     // all 4 reads complete for this frame

    // =========================================================================
    // State machine
    // =========================================================================
    localparam [2:0]
        IDLE        = 3'd0,
        WRITE_REQ   = 3'd1,
        WRITE_ACK   = 3'd2,
        WRITE_WAIT  = 3'd3,
        READ_REQ    = 3'd4,
        READ_ACK    = 3'd5,
        READ_WAIT   = 3'd6;

    reg [2:0] state;

    // Address of the word currently being written (tracks sd_count before increment)
    reg [1:0] write_idx;

    always @(posedge clk_mem or negedge resetn) begin
        if (!resetn) begin
            state        <= IDLE;
            mem_addr_out <= 28'd0;
            mem_data_out <= 64'd0;
            wflag        <= 1'b0;
            rflag        <= 1'b0;
            read_idx     <= 2'd0;
            write_idx    <= 2'd0;
            frame_ready  <= 1'b0;
            word_buf[0]  <= 32'h0;
            word_buf[1]  <= 32'h0;
            word_buf[2]  <= 32'h0;
            word_buf[3]  <= 32'h0;
            display_words <= 128'h0;
            display_valid <= 1'b0;
        end else begin
            // Default: de-assert strobes each cycle unless held below
            frame_ready <= 1'b0;

            case (state)

                // ------------------------------------------------------------------
                IDLE: begin
                    wflag <= 1'b0;
                    rflag <= 1'b0;

                    if (sd_pending && !all_written) begin
                        // Prioritise writing new SD words into DDR2
                        mem_addr_out <= {26'd0, write_idx, 1'b0};  // byte addr = word_index * 8
                        mem_data_out <= {32'h0000_0000, sd_word_lat};
                        wflag        <= 1'b1;
                        state        <= WRITE_REQ;

                    end else if (all_written) begin
                        // Continuously read all 4 words and refresh display
                        mem_addr_out <= {26'd0, read_idx, 1'b0};   // byte addr = read_index * 8
                        rflag        <= 1'b1;
                        state        <= READ_REQ;
                    end
                end

                // ------------------------------------------------------------------
                // WRITE path
                // ------------------------------------------------------------------
                WRITE_REQ: begin
                    // Hold wflag; wait for writefini to go HIGH
                    if (writefini) begin
                        wflag <= 1'b0;
                        state <= WRITE_ACK;
                    end
                end

                WRITE_ACK: begin
                    // Wait for writefini to go LOW (handshake complete)
                    if (!writefini) begin
                        write_idx <= write_idx + 2'd1;
                        state     <= WRITE_WAIT;
                    end
                end

                WRITE_WAIT: begin
                    // One idle cycle before next transaction
                    state <= IDLE;
                end

                // ------------------------------------------------------------------
                // READ path - read one word per pass; after 4 reads assemble frame
                // ------------------------------------------------------------------
                READ_REQ: begin
                    // Hold rflag; wait for readfini to go HIGH
                    if (readfini) begin
                        rflag <= 1'b0;
                        state <= READ_ACK;
                    end
                end

                READ_ACK: begin
                    // Latch the returned 32-bit word (upper half is zero-padded)
                    word_buf[read_idx] <= mem_data_in[31:0];
                    state <= READ_WAIT;
                end

                READ_WAIT: begin
                    // Wait for readfini to go LOW
                    if (!readfini) begin
                        if (read_idx == 2'd3) begin
                            // All 4 words received - publish to VGA
                            read_idx     <= 2'd0;
                            frame_ready  <= 1'b1;
                        end else begin
                            read_idx <= read_idx + 2'd1;
                        end
                        state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase

            // ------------------------------------------------------------------
            // Latch completed frame into display_words (stable for VGA CDC)
            // ------------------------------------------------------------------
            if (frame_ready) begin
                display_words <= { word_buf[0],   // sprite 0 -> [127:96]
                                   word_buf[1],   // sprite 1 -> [ 95:64]
                                   word_buf[2],   // sprite 2 -> [ 63:32]
                                   word_buf[3] }; // sprite 3 -> [ 31: 0]
                display_valid <= 1'b1;
            end
        end
    end

endmodule
