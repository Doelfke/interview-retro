#!/usr/bin/env node

import { execSync } from "child_process";
import { pathToFileURL } from "url";

const FIX_CHECKS_PROMPT_PREFIX = "Fix these issues:";
const VSCODE_FORMAT = "vscode";
const ANSI_REGEX = /\u001B\[[0-?]*[ -/]*[@-~]/g;

const CHECKS = [
  {
    title: "Lint",
    command: "uv run ruff check backend/",
  },
  {
    title: "Type errors",
    command: "uv run pyright backend/",
  },
];

function runCheck(command) {
  const stripAnsi = (text) => text.replace(ANSI_REGEX, "");

  try {
    const output = execSync(command, { encoding: "utf8", stdio: "pipe" });
    return { output: stripAnsi(output), failed: false };
  } catch (err) {
    const output = `${err.stdout ?? ""}${err.stderr ?? ""}`;
    return { output: stripAnsi(output), failed: true };
  }
}

function createStopBlockResponse(reason, isVsCode = false) {
  const fullReason = `${FIX_CHECKS_PROMPT_PREFIX}\n\n${reason}`;

  if (isVsCode) {
    return {
      hookSpecificOutput: {
        hookEventName: "Stop",
        decision: "block",
        reason: fullReason,
      },
    };
  }

  return {
    decision: "block",
    reason: fullReason,
  };
}

function hasChangedFiles() {
  try {
    const output = execSync("git status --porcelain", {
      encoding: "utf8",
      stdio: "pipe",
    }).trim();
    return output.length > 0;
  } catch {
    // Not a git repo or git unavailable — run checks anyway
    return true;
  }
}

export function main(argv = process.argv) {
  const isVsCode = argv.some((arg) => arg === `--format=${VSCODE_FORMAT}`);

  if (!hasChangedFiles()) return;

  for (const check of CHECKS) {
    const { output, failed } = runCheck(check.command);
    if (failed) {
      console.log(
        JSON.stringify(
          createStopBlockResponse(`${check.title}:\n\n${output}`, isVsCode),
          null,
          0,
        ),
      );
      return;
    }
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main();
}
