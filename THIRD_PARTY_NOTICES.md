# Third-Party Notices

## MemBase Mem0 prompt design

RSIMem's local Mem0-flat prompt artifacts are rewritten from the design of
`FACT_RETRIEVAL_PROMPT` and `DEFAULT_UPDATE_MEMORY_PROMPT` in:

- Repository: `https://github.com/zjunlp/MemBase`
- Commit: `d2aca6c7abcb1d67b331586cb834495d037fa3a6`
- Path: `membase/baselines/mem0/configs/prompts.py`
- Source file SHA-256: `bf92192da5033a6793531d55d87945d8bf8728517e0c0c0690e83cb0e0042849`
- License: MIT
- Copyright: Copyright (c) 2026 ZJUNLP

The local templates retain the memory-construction ideas, but exclude MemBase
answering instructions, dynamic wall-clock text, and instructions to claim
that remembered information came from public internet sources. Local template
and modification digests are exposed through each `PromptArtifact` manifest
record.

The upstream work is provided under the MIT License:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
