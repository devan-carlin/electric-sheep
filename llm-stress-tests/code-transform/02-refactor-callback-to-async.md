# Refactor Callbacks → Async/Await

**Category:** Code transformation
**Target:** Understanding control flow, error propagation, concurrency patterns

---

## Prompt

Refactor the following Node.js code from callback-based to async/await. The code implements a simple file processing pipeline that reads config, fetches data, transforms it, and writes output.

**Source (callbacks):**

```javascript
const fs = require('fs');
const http = require('http');
const path = require('path');

function processPipeline(configPath, callback) {
    // Step 1: Read config
    fs.readFile(configPath, 'utf8', (err, configData) => {
        if (err) return callback(err);
        
        let config;
        try {
            config = JSON.parse(configData);
        } catch (e) {
            return callback(new Error('Invalid config JSON: ' + e.message));
        }
        
        // Step 2: Fetch data from each source
        const sources = config.sources || [];
        const allData = [];
        let completed = 0;
        let failed = false;
        
        if (sources.length === 0) {
            return callback(new Error('No sources configured'));
        }
        
        sources.forEach((source) => {
            http.get(source.url, (res) => {
                if (failed) return;
                
                if (res.statusCode !== 200) {
                    failed = true;
                    return callback(new Error(`HTTP ${res.statusCode} from ${source.url}`));
                }
                
                let body = '';
                res.on('data', (chunk) => body += chunk);
                res.on('end', () => {
                    if (failed) return;
                    
                    try {
                        const data = JSON.parse(body);
                        // Step 3: Transform
                        const transformed = data.items.map(item => ({
                            id: item.id,
                            name: item.name.toUpperCase(),
                            score: item.raw_score * (source.weight || 1),
                            source: source.name
                        })).filter(item => item.score >= (config.threshold || 0));
                        
                        allData.push(...transformed);
                        completed++;
                        
                        if (completed === sources.length) {
                            // Step 4: Sort and write
                            allData.sort((a, b) => b.score - a.score);
                            
                            const outputPath = path.join(config.output_dir || '.', 'results.json');
                            fs.writeFile(outputPath, JSON.stringify(allData, null, 2), (writeErr) => {
                                if (writeErr) return callback(writeErr);
                                callback(null, { written: allData.length, path: outputPath });
                            });
                        }
                    } catch (e) {
                        if (!failed) {
                            failed = true;
                            callback(new Error('Transform error: ' + e.message));
                        }
                    }
                });
            }).on('error', (e) => {
                if (!failed) {
                    failed = true;
                    callback(new Error(`Request failed: ${e.message}`));
                }
            });
        });
    });
}

module.exports = { processPipeline };
```

**Requirements for the async/await version:**

- Use `async/await` throughout — zero callbacks
- Use `fs.promises` and `http.get` with promise wrapper
- Parallel fetch with `Promise.allSettled` (one failed source shouldn't kill the whole pipeline)
- Proper error handling with descriptive messages that include which source failed
- Retry logic: retry failed sources up to 2 times with 1-second delay
- Timeout: each source request should timeout after 10 seconds
- Return a summary object: `{ success: N, failed: N, written: N, errors: [...] }`
- Include a test file that mocks `http.get` and verifies behavior

**Constraints:**

- Node.js 18+
- No external dependencies (stdlib only)
- Must handle: empty config, no sources, all sources fail, partial failure
- Include JSDoc comments on the main function

Produce all files with complete working code. No placeholders, no TODOs.
