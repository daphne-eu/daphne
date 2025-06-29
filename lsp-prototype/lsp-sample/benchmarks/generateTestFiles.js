const fs = require('fs');
const path = require('path');

const generateFile = (lines, filename) => {
  let content = '';
  for (let i = 0; i < lines; i++) {
    content += `VAR X${i} := ${i};\n`;  // Trigger diagnostics with UPPERCASE
  }

  const fullPath = path.join(__dirname, filename);
  fs.writeFileSync(fullPath, content);
  console.log(`✅ Generated: ${filename} (${lines} lines)`);
};

// Generate test files
generateFile(50, 'test_50.daphne');
generateFile(500, 'test_500.daphne');
generateFile(5000, 'test_5000.daphne');
