import struct
import os
import sys

# Force UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import tiktoken

def prepare_data(input_txt="input.txt", output_bin="train_tokens.bin"):
    enc = tiktoken.get_encoding("gpt2")

    if not os.path.exists(input_txt):
        print(f"Creating sample training dataset in {input_txt}...")
        sample_code = """
// --- C / C++ / Zig / Rust / Python / Java Sample Code ---

#include <stdio.h>
int main() {
    printf("Hello from C!\\n");
    return 0;
}

const std = @import("std");
pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Hello from Zig!\\n", .{});
}

fn fibonacci(n: u32) u32 {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

def solve_problem(items):
    total = 0
    for x in items:
        if x % 2 == 0:
            total += x * 2
        else:
            total += x
    return total

class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        res = a + b
        self.history.append(('add', a, b, res))
        return res
""" * 50
        with open(input_txt, "w", encoding="utf-8") as f:
            f.write(sample_code)

    print(f"Reading {input_txt}...")
    with open(input_txt, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokens = enc.encode(text)
    print(f"Encoded {len(tokens)} tokens.")

    print(f"Writing to {output_bin}...")
    with open(output_bin, "wb") as f:
        # Header: count of tokens (int32)
        f.write(struct.pack("i", len(tokens)))
        for t in tokens:
            f.write(struct.pack("i", t))

    size_kb = os.path.getsize(output_bin) / 1024
    print(f"Done! {output_bin} created ({size_kb:.1f} KB).")

if __name__ == "__main__":
    prepare_data()
