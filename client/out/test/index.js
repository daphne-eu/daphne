"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.run = run;
/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */
const path = require("path");
const Mocha = require("mocha");
const glob_1 = require("glob");
function run() {
    // Create the mocha test
    const mocha = new Mocha({
        ui: 'tdd',
        color: true
    });
    mocha.timeout(100000);
    const testsRoot = __dirname;
    return glob_1.glob.glob('**.test.js', { cwd: testsRoot }).then(async (files) => {
        // Add files to the test suite
        files.forEach(f => mocha.addFile(path.resolve(testsRoot, f)));
        try {
            // Run the mocha test
            await new Promise((resolve, reject) => {
                mocha.run(failures => {
                    if (failures > 0) {
                        reject(`${failures} tests failed.`);
                    }
                    else {
                        resolve();
                    }
                });
            });
        }
        catch (err) {
            console.error(err);
            throw err;
        }
    });
}
//# sourceMappingURL=index.js.map