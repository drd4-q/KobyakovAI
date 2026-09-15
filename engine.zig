const std = @import("std");
const math = std.math;

// ============================================================
// Windows API (Win32) для нативного I/O без libc
// ============================================================
const INVALID_HANDLE_VALUE: *anyopaque = @ptrFromInt(~@as(usize, 0));
const GENERIC_READ: u32 = 0x80000000;
const FILE_SHARE_READ: u32 = 1;
const OPEN_EXISTING: u32 = 3;
const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
const STD_OUTPUT_HANDLE: u32 = @bitCast(@as(i32, -11));

extern "kernel32" fn CreateFileA(
    lpFileName: [*:0]const u8,
    dwDesiredAccess: u32,
    dwShareMode: u32,
    lpSecurityAttributes: ?*anyopaque,
    dwCreationDisposition: u32,
    dwFlagsAndAttributes: u32,
    hTemplateFile: ?*anyopaque,
) callconv(.winapi) *anyopaque;

extern "kernel32" fn GetFileSize(
    hFile: *anyopaque,
    lpFileSizeHigh: ?*u32,
) callconv(.winapi) u32;

extern "kernel32" fn ReadFile(
    hFile: *anyopaque,
    lpBuffer: [*]u8,
    nNumberOfBytesToRead: u32,
    lpNumberOfBytesRead: ?*u32,
    lpOverlapped: ?*anyopaque,
) callconv(.winapi) i32;

extern "kernel32" fn CloseHandle(
    hObject: *anyopaque,
) callconv(.winapi) i32;

extern "kernel32" fn GetStdHandle(
    nStdHandle: u32,
) callconv(.winapi) *anyopaque;

extern "kernel32" fn WriteFile(
    hFile: *anyopaque,
    lpBuffer: [*]const u8,
    nNumberOfBytesToWrite: u32,
    lpNumberOfBytesWritten: ?*u32,
    lpOverlapped: ?*anyopaque,
) callconv(.winapi) i32;

fn printStr(stdout: *anyopaque, s: []const u8) void {
    var written: u32 = 0;
    _ = WriteFile(stdout, s.ptr, @intCast(s.len), &written, null);
}

// ============================================================
// Конфигурация модели
// ============================================================
pub const Config = extern struct {
    vocab_size: i32,
    block_size: i32,
    n_embd: i32,
    n_layer: i32,
    n_head: i32,
    num_experts: i32,
    top_k: i32,
};

pub const ExpertWeights = struct {
    w1: []const f32,
    b1: []const f32,
    w2: []const f32,
    b2: []const f32,
};

pub const LayerWeights = struct {
    sa_ln_w: []const f32,
    sa_ln_b: []const f32,
    sa_wq: []const f32,
    sa_wk: []const f32,
    sa_wv: []const f32,
    sa_wo: []const f32,
    moe_ln_w: []const f32,
    moe_ln_b: []const f32,
    moe_gate_w: []const f32,
    moe_gate_b: []const f32,
    experts: []ExpertWeights,
};

pub const MoEModel = struct {
    config: Config,
    raw_data: []f32,
    token_embedding_table: []const f32,
    position_embedding_table: []const f32,
    layers: []LayerWeights,
    ln_f_w: []const f32,
    ln_f_b: []const f32,
    lm_head: []const f32,
};

pub const RunState = struct {
    x: []f32,
    xb: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    att: []f32,
    key_cache: []f32,
    val_cache: []f32,
    gate_logits: []f32,
    exp_h: []f32,
    exp_out: []f32,
    moe_acc: []f32,
    logits: []f32,
};

// ============================================================
// Базовая математика
// ============================================================

fn layernorm(out: []f32, x: []const f32, w: []const f32, b: []const f32) void {
    const size = x.len;
    var mean: f32 = 0.0;
    for (x) |val| mean += val;
    mean /= @as(f32, @floatFromInt(size));

    var variance: f32 = 0.0;
    for (x) |val| {
        const diff = val - mean;
        variance += diff * diff;
    }
    variance /= @as(f32, @floatFromInt(size));
    const inv_std: f32 = 1.0 / @sqrt(variance + 1e-5);

    for (0..size) |i| {
        out[i] = ((x[i] - mean) * inv_std) * w[i] + b[i];
    }
}

fn matmul(out: []f32, x: []const f32, w: []const f32, d_in: usize, d_out: usize) void {
    for (0..d_out) |i| {
        var val: f32 = 0.0;
        const row_start = i * d_in;
        for (0..d_in) |j| {
            val += w[row_start + j] * x[j];
        }
        out[i] = val;
    }
}

fn softmax(x: []f32) void {
    var max_val = x[0];
    for (x[1..]) |val| {
        if (val > max_val) max_val = val;
    }
    var sum: f32 = 0.0;
    for (x) |*val| {
        val.* = @exp(val.* - max_val);
        sum += val.*;
    }
    const inv_sum = 1.0 / (sum + 1e-8);
    for (x) |*val| {
        val.* *= inv_sum;
    }
}

fn gelu(x: []f32) void {
    const k: f32 = @sqrt(2.0 / math.pi);
    for (x) |*val| {
        const xi = val.*;
        const cube = 0.044715 * xi * xi * xi;
        val.* = 0.5 * xi * (1.0 + math.tanh(k * (xi + cube)));
    }
}

// ============================================================
// Прямой проход (Forward Pass)
// ============================================================

fn forward(model: *const MoEModel, s: *RunState, token: usize, pos: usize) []const f32 {
    const p = model.config;
    const d: usize = @intCast(p.n_embd);
    const head_size: usize = d / @as(usize, @intCast(p.n_head));
    const n_head: usize = @intCast(p.n_head);
    const n_layer: usize = @intCast(p.n_layer);
    const block_size: usize = @intCast(p.block_size);
    const num_experts: usize = @intCast(p.num_experts);
    const top_k: usize = @intCast(p.top_k);

    // 1. Embeddings
    const tok_vec = model.token_embedding_table[token * d .. (token + 1) * d];
    const pos_idx = pos % block_size;
    const pos_vec = model.position_embedding_table[pos_idx * d .. (pos_idx + 1) * d];
    for (0..d) |i| {
        s.x[i] = tok_vec[i] + pos_vec[i];
    }

    // 2. Transformer layers
    for (0..n_layer) |l| {
        const lw = &model.layers[l];

        // Self-Attention LN
        layernorm(s.xb, s.x, lw.sa_ln_w, lw.sa_ln_b);

        // Q, K, V
        matmul(s.q, s.xb, lw.sa_wq, d, d);
        matmul(s.k, s.xb, lw.sa_wk, d, d);
        matmul(s.v, s.xb, lw.sa_wv, d, d);

        // Cache update
        const cache_offset = (l * block_size + pos) * d;
        @memcpy(s.key_cache[cache_offset .. cache_offset + d], s.k);
        @memcpy(s.val_cache[cache_offset .. cache_offset + d], s.v);

        @memset(s.xb, 0.0);

        for (0..n_head) |h| {
            const q_head = s.q[h * head_size .. (h + 1) * head_size];
            const att_head = s.att[h * block_size .. (h + 1) * block_size];

            for (0..pos + 1) |t| {
                const k_offset = (l * block_size + t) * d + h * head_size;
                const k_head = s.key_cache[k_offset .. k_offset + head_size];
                var score: f32 = 0.0;
                for (0..head_size) |i| {
                    score += q_head[i] * k_head[i];
                }
                score /= @sqrt(@as(f32, @floatFromInt(head_size)));
                att_head[t] = score;
            }

            softmax(att_head[0 .. pos + 1]);

            for (0..pos + 1) |t| {
                const a = att_head[t];
                const v_offset = (l * block_size + t) * d + h * head_size;
                const v_head = s.val_cache[v_offset .. v_offset + head_size];
                for (0..head_size) |i| {
                    s.xb[h * head_size + i] += a * v_head[i];
                }
            }
        }

        // Wo
        matmul(s.q, s.xb, lw.sa_wo, d, d);
        for (0..d) |i| {
            s.x[i] += s.q[i]; // residual
        }

        // --- Sparse MoE ---
        layernorm(s.xb, s.x, lw.moe_ln_w, lw.moe_ln_b);

        // Router gate
        matmul(s.gate_logits[0..num_experts], s.xb, lw.moe_gate_w, d, num_experts);
        for (0..num_experts) |e| {
            s.gate_logits[e] += lw.moe_gate_b[e];
        }
        softmax(s.gate_logits[0..num_experts]);

        // Top-K experts
        var top_indices: [8]usize = undefined;
        var top_weights: [8]f32 = undefined;
        for (0..top_k) |k_idx| {
            var best_idx: usize = 0;
            var best_val: f32 = -1e9;
            for (0..num_experts) |e| {
                var already = false;
                for (0..k_idx) |prev| {
                    if (top_indices[prev] == e) already = true;
                }
                if (!already and s.gate_logits[e] > best_val) {
                    best_val = s.gate_logits[e];
                    best_idx = e;
                }
            }
            top_indices[k_idx] = best_idx;
            top_weights[k_idx] = best_val;
        }

        var w_sum: f32 = 0.0;
        for (0..top_k) |k_idx| w_sum += top_weights[k_idx];
        for (0..top_k) |k_idx| top_weights[k_idx] /= (w_sum + 1e-8);

        @memset(s.moe_acc, 0.0);
        for (0..top_k) |k_idx| {
            const exp_id = top_indices[k_idx];
            const exp_w = top_weights[k_idx];
            const ew = &lw.experts[exp_id];

            matmul(s.exp_h, s.xb, ew.w1, d, 4 * d);
            for (0..4 * d) |i| s.exp_h[i] += ew.b1[i];
            gelu(s.exp_h);

            matmul(s.exp_out, s.exp_h, ew.w2, 4 * d, d);
            for (0..d) |i| {
                s.moe_acc[i] += exp_w * (s.exp_out[i] + ew.b2[i]);
            }
        }

        for (0..d) |i| {
            s.x[i] += s.moe_acc[i]; // residual
        }
    }

    // 3. Final LN
    layernorm(s.xb, s.x, model.ln_f_w, model.ln_f_b);

    // 4. LM Head
    const vocab_size: usize = @intCast(p.vocab_size);
    matmul(s.logits, s.xb, model.lm_head, d, vocab_size);

    return s.logits;
}

// ============================================================
// Токенизатор
// ============================================================

pub const Tokenizer = struct {
    vocab: [][]const u8,
    allocator: std.mem.Allocator,

    pub fn load(allocator: std.mem.Allocator, path: [*:0]const u8) ?Tokenizer {
        const handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, null, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, null);
        if (handle == INVALID_HANDLE_VALUE) return null;
        defer _ = CloseHandle(handle);

        var vocab_size: i32 = 0;
        var bytes_read: u32 = 0;
        if (ReadFile(handle, @ptrCast(&vocab_size), 4, &bytes_read, null) == 0 or bytes_read != 4) return null;

        const u_size: usize = @intCast(vocab_size);
        const vocab = allocator.alloc([]const u8, u_size) catch return null;

        for (0..u_size) |i| {
            var len: i32 = 0;
            _ = ReadFile(handle, @ptrCast(&len), 4, &bytes_read, null);
            if (len > 0) {
                const u_len: usize = @intCast(len);
                const buf = allocator.alloc(u8, u_len) catch return null;
                _ = ReadFile(handle, buf.ptr, @intCast(u_len), &bytes_read, null);
                vocab[i] = buf;
            } else {
                vocab[i] = "";
            }
        }

        return Tokenizer{
            .vocab = vocab,
            .allocator = allocator,
        };
    }
};

fn sampleToken(logits: []const f32, seed: *u64, temp: f32) usize {
    if (temp <= 0.01) {
        var best_i: usize = 0;
        var best_v = logits[0];
        for (logits[1..], 1..) |v, i| {
            if (v > best_v) {
                best_v = v;
                best_i = i;
            }
        }
        return best_i;
    }

    var max_v = logits[0];
    for (logits[1..]) |v| {
        if (v > max_v) max_v = v;
    }

    var sum: f32 = 0.0;
    for (logits) |v| {
        sum += @exp((v - max_v) / temp);
    }

    // Xorshift RNG
    seed.* ^= seed.* << 13;
    seed.* ^= seed.* >> 7;
    seed.* ^= seed.* << 17;
    const r = @as(f32, @floatFromInt(seed.* % 1000000)) / 1000000.0 * sum;

    var cdf: f32 = 0.0;
    for (logits, 0..) |v, i| {
        cdf += @exp((v - max_v) / temp);
        if (r <= cdf) return i;
    }
    return logits.len - 1;
}

// ============================================================
// Точка входа (Main)
// ============================================================

pub fn main() !void {
    const stdout = GetStdHandle(STD_OUTPUT_HANDLE);
    const allocator = std.heap.page_allocator;

    printStr(stdout, "============================================================\r\n");
    printStr(stdout, "  ⚡ NATIVE ZIG MoE TRANSFORMER ENGINE (Чистый Zig)\r\n");
    printStr(stdout, "============================================================\r\n");

    const handle = CreateFileA("model.bin", GENERIC_READ, FILE_SHARE_READ, null, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, null);
    if (handle == INVALID_HANDLE_VALUE) {
        printStr(stdout, "Ошибка: не удалось открыть model.bin\r\n");
        return;
    }
    defer _ = CloseHandle(handle);

    const file_size = GetFileSize(handle, null);
    const f32_count = file_size / @sizeOf(f32);
    const f32_buffer = try allocator.alloc(f32, f32_count);
    defer allocator.free(f32_buffer);

    const byte_slice = std.mem.sliceAsBytes(f32_buffer);
    var bytes_read: u32 = 0;
    _ = ReadFile(handle, byte_slice.ptr, file_size, &bytes_read, null);

    const config = @as(*const Config, @ptrCast(byte_slice[0..@sizeOf(Config)].ptr)).*;
    std.debug.print("[Zig Engine] Config: vocab={d}, block={d}, embd={d}, layers={d}, heads={d}, experts={d}, top_k={d}\n", .{
        config.vocab_size, config.block_size, config.n_embd, config.n_layer, config.n_head, config.num_experts, config.top_k,
    });

    const header_f32_offset = @sizeOf(Config) / @sizeOf(f32);
    var ptr = f32_buffer[header_f32_offset..].ptr;

    const d: usize = @intCast(config.n_embd);
    const vocab_size: usize = @intCast(config.vocab_size);
    const block_size: usize = @intCast(config.block_size);
    const n_layer: usize = @intCast(config.n_layer);
    const num_experts: usize = @intCast(config.num_experts);

    const tok_emb = ptr[0 .. vocab_size * d]; ptr += vocab_size * d;
    const pos_emb = ptr[0 .. block_size * d]; ptr += block_size * d;

    const layers = try allocator.alloc(LayerWeights, n_layer);
    defer {
        for (layers) |lw| allocator.free(lw.experts);
        allocator.free(layers);
    }

    for (0..n_layer) |l| {
        var lw = &layers[l];
        lw.sa_ln_w = ptr[0..d]; ptr += d;
        lw.sa_ln_b = ptr[0..d]; ptr += d;
        lw.sa_wq = ptr[0 .. d * d]; ptr += d * d;
        lw.sa_wk = ptr[0 .. d * d]; ptr += d * d;
        lw.sa_wv = ptr[0 .. d * d]; ptr += d * d;
        lw.sa_wo = ptr[0 .. d * d]; ptr += d * d;
        lw.moe_ln_w = ptr[0..d]; ptr += d;
        lw.moe_ln_b = ptr[0..d]; ptr += d;
        lw.moe_gate_w = ptr[0 .. num_experts * d]; ptr += num_experts * d;
        lw.moe_gate_b = ptr[0..num_experts]; ptr += num_experts;

        lw.experts = try allocator.alloc(ExpertWeights, num_experts);
        for (0..num_experts) |e| {
            var ew = &lw.experts[e];
            ew.w1 = ptr[0 .. 4 * d * d]; ptr += 4 * d * d;
            ew.b1 = ptr[0 .. 4 * d]; ptr += 4 * d;
            ew.w2 = ptr[0 .. d * 4 * d]; ptr += d * 4 * d;
            ew.b2 = ptr[0..d]; ptr += d;
        }
    }

    const ln_f_w = ptr[0..d]; ptr += d;
    const ln_f_b = ptr[0..d]; ptr += d;
    const lm_head = ptr[0 .. vocab_size * d];

    const model = MoEModel{
        .config = config,
        .raw_data = f32_buffer,
        .token_embedding_table = tok_emb,
        .position_embedding_table = pos_emb,
        .layers = layers,
        .ln_f_w = ln_f_w,
        .ln_f_b = ln_f_b,
        .lm_head = lm_head,
    };

    var state = RunState{
        .x = try allocator.alloc(f32, d),
        .xb = try allocator.alloc(f32, d),
        .q = try allocator.alloc(f32, d),
        .k = try allocator.alloc(f32, d),
        .v = try allocator.alloc(f32, d),
        .att = try allocator.alloc(f32, @as(usize, @intCast(config.n_head)) * block_size),
        .key_cache = try allocator.alloc(f32, n_layer * block_size * d),
        .val_cache = try allocator.alloc(f32, n_layer * block_size * d),
        .gate_logits = try allocator.alloc(f32, num_experts),
        .exp_h = try allocator.alloc(f32, 4 * d),
        .exp_out = try allocator.alloc(f32, d),
        .moe_acc = try allocator.alloc(f32, d),
        .logits = try allocator.alloc(f32, vocab_size),
    };
    defer {
        allocator.free(state.x);
        allocator.free(state.xb);
        allocator.free(state.q);
        allocator.free(state.k);
        allocator.free(state.v);
        allocator.free(state.att);
        allocator.free(state.key_cache);
        allocator.free(state.val_cache);
        allocator.free(state.gate_logits);
        allocator.free(state.exp_h);
        allocator.free(state.exp_out);
        allocator.free(state.moe_acc);
        allocator.free(state.logits);
    }

    var tok = Tokenizer.load(allocator, "tokenizer.bin");
    defer if (tok) |*t| {
        for (t.vocab) |v| if (v.len > 0) t.allocator.free(v);
        t.allocator.free(t.vocab);
    };

    var seed: u64 = 123456789;

    printStr(stdout, "\r\nСтартовый промпт: /* C/Zig MoE AI */\r\n");
    printStr(stdout, "🤖 [MoE Zig-Engine]: ");

    var current_token: usize = 15496;
    const max_tokens: usize = 64;

    for (0..max_tokens) |pos| {
        const logits = forward(&model, &state, current_token, pos);
        current_token = sampleToken(logits, &seed, 0.8);

        if (tok) |t| {
            if (current_token < t.vocab.len) {
                printStr(stdout, t.vocab[current_token]);
            } else {
                printStr(stdout, ".");
            }
        } else {
            printStr(stdout, ".");
        }
    }

    printStr(stdout, "\r\n\r\n------------------------------------------------------------\r\n");
    printStr(stdout, "Генерация на нативном Zig завершена успешно!\r\n");
    printStr(stdout, "------------------------------------------------------------\r\n");
}
