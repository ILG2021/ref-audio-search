import { indexDirectory } from "./indexer.js";
import { closeDb } from "./db.js";
import { modelProvider } from "./model-provider.js";

const [, , command, directory] = process.argv;
if (command !== "index" || !directory) {
  console.error("Usage: npm run index -- <audio-directory>");
  process.exit(1);
}

try {
  const result = await indexDirectory(directory, ({ completed, total, file }) => {
    process.stdout.write(`\r${completed}/${total} ${file.slice(-60).padEnd(60)}`);
  });
  console.log(`\nIndexed ${result.indexed}, skipped unchanged ${result.skipped}, total ${result.total}`);
  if (result.errors.length) {
    console.error(result.errors);
    process.exitCode = 2;
  }
} finally {
  modelProvider.stop();
  closeDb();
}
