# pylyzer_wasm

Wasm wrapper for pylyzer.

## Usage

```ts
import { Analyzer } from 'pylyzer_wasm';

const analyzer = new Analyzer();
const errors = analyzer.check('print("Hello, World!")');
const locals = analyzer.dir();
```
