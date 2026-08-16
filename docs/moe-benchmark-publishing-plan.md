# MoE Benchmark Publishing Plan

> **Project**: Document and share the MoE TopK kernel patch + quality paradox findings
> **Date**: August 14, 2026
> **Status**: Planning phase

---

## Overview

We have a complete technical story with three compelling angles:
1. **The kernel patch hunt** — debugging two SYCL kernel files to enable TopK=16/32/64
2. **The performance scaling curve** — non-linear slowdown as experts increase (7% → 14% → 26%)
3. **The quality paradox** — activating 8× more experts actually *hurt* output quality

This plan covers preparing the content, publishing across platforms, and maximizing reach.

---

## Phase 1: GitHub Preparation (Day 0)

### 1.1 Commit the guide
```bash
cd /home/dc/electric-sheep
git add docs/guides/moe-topk-kernel-patch.md
git commit -m "docs: add comprehensive MoE TopK kernel patch guide with benchmark results"
```

### 1.2 Create a benchmarking README
Create `/home/dc/electric-sheep/benchmarking/README.md` with:
- Hardware setup (4× Arc B70, TP=4)
- Model info (Huihui-, Qwen3.5 MoE)
- Quick summary table (performance + quality)
- Links to full guide and results

### 1.3 Clean up benchmark results
- Ensure `results/SUMMARY.md` is current
- Ensure `results/outputs/` has all 32 output files (4 variants × 8 prompts)
- Add `results/COMPARISON-REPORT.md` if it doesn't exist

### 1.4 Tag a release
```bash
cd /home/dc/electric-sheep
git tag -a v1.0-moe-benchmark -m "MoE TopK kernel patch + benchmark results"
git push origin main --tags
```

### 1.5 Prepare GitHub for public viewing
- [ ] Remove any sensitive paths/credentials from logs
- [ ] Check `.gitignore` doesn't expose secrets
- [ ] Consider making the repo public (or fork to a public org)
- [ ] Add license (MIT or Apache 2.0)

---

## Phase 2: Content Creation (Day 1-2)

### 2.1 Medium Article Draft

**Working title**: "We Patched vLLM to Route 64 Experts — Here's What Happened"

**Structure**:
1. **Hook**: "I thought more experts = better answers. I was wrong."
2. **The setup**: 4× Intel Arc B70 homelab, TP=4, MoE model
3. **The problem**: vLLM only supported TopK up to 10
4. **The kernel hunt**: Patched `remap_hidden_states.cpp`, still failed, found `moe_gather.cpp`
5. **The benchmark**: All 4 variants now work — here are the numbers
6. **The paradox**: Quality dropped at TopK=64 (math error, shorter outputs)
7. **The analysis**: Why more experts hurt (expert noise, router confidence, fixed compute budget)
8. **The takeaway**: There's a sweet spot (8-16 experts) — not "more is better"
9. **Links**: GitHub repo, full guide, raw data

**Tone**: Conversational but technical. Write like you're explaining to a smart engineer over coffee.

**Length**: ~2,500-3,500 words

### 2.2 Charts and Visuals

Create 4 charts (use Python/matplotlib or hand-draw in Figma):

| Chart | Data | Purpose |
|-------|------|---------|
| **Performance scaling** | TopK vs throughput (83 → 77 → 66 → 49 tok/s) | Show non-linear slowdown |
| **Quality scores** | TopK vs quality (4.0 → 4.1 → 4.0 → 3.4) | Show quality cliff |
| **Speed vs quality** | Scatter plot (throughput on x-axis, quality on y-axis) | Show tradeoff |
| **Output size comparison** | Bytes per prompt across variants | Show top-64 produces shorter outputs |

### 2.3 X/Twitter Thread Draft

**Tweet 1 (Hook)**:
> We activated 8× more experts in a MoE model. The model got worse at math.
> 
> Here's what happened when I patched vLLM to route 64 experts per token on 4× Intel Arc B70s. 🧵

**Tweet 2 (Setup)**:
> Hardware: 4× Intel Arc Pro B70 (128GB VRAM total)
> Model: Qwen3.5 MoE (256 experts, 40 layers)
> vLLM: TP=4, FlashAttention v2
> 
> Goal: Compare TopK=8, 16, 32, 64 expert routing

**Tweet 3 (The patch)**:
> vLLM only supported TopK up to 10. Had to patch two SYCL kernel files:
> 
> - remap_hidden_states.cpp (expert routing)
> - moe_gather.cpp (expert aggregation)
> 
> 9 lines of C++ added. 30-minute SYCL rebuild.

**Tweet 4 (Performance)**:
> Results:
> - TopK=8:  83 tok/s (baseline)
> - TopK=16: 77 tok/s (-7%)
> - TopK=32: 66 tok/s (-20%)
> - TopK=64: 49 tok/s (-41%)
> 
> The slowdown is non-linear. Each doubling hurts more.

**Tweet 5 (Quality)**:
> Quality scores (1-5 scale):
> - TopK=8:  4.0
> - TopK=16: 4.1
> - TopK=32: 4.0
> - TopK=64: 3.4 ← quality cliff
> 
> TopK=64 computed 0.06/12 = 0.05 (should be 0.005). That's not "different reasoning" — that's a math error.

**Tweet 6 (The paradox)**:
> The MoE Routing Quality Paradox:
> 
> More experts ≠ better answers. Beyond ~16 experts, you're averaging in noise from low-confidence specialists. The router was trained for ~8 experts. Forcing 64 dilutes the signal.
> 
> There's a sweet spot. It's not "more is better."

**Tweet 7 (CTA)**:
> Full article with kernel diffs, benchmark data, and analysis:
> [link to Medium]
> 
> Code and patches on GitHub:
> [link to GitHub]

### 2.4 LinkedIn Post Draft

**Structure**:
- Professional summary of the work
- Key finding (quality paradox)
- Engineering process (kernel patching, benchmarking)
- Link to full article
- Tags: #AI #MachineLearning #IntelGPU #OpenSource #MoE

**Tone**: Professional but not dry. Show expertise without being academic.

### 2.5 Reddit Posts (3 variants)

**r/MachineLearning**:
- Title: "MoE Routing Quality Paradox: Activating 64 Experts Hurt Output Quality (Benchmark on 4× Intel Arc B70)"
- Focus: Research finding, quality analysis, methodology
- Link to Medium article

**r/localLLaMA**:
- Title: "Patched vLLM to support TopK=64 on Intel Arc B70s — here are the benchmark results"
- Focus: Practical stuff, kernel patches, performance numbers
- Link to GitHub + Medium

**r/intel**:
- Title: "Benchmarking MoE models on 4× Intel Arc Pro B70 (TP=4) — performance and quality analysis"
- Focus: Hardware performance, Intel GPU capabilities
- Link to Medium article

---

## Phase 3: Publishing Sequence (Day 3-5)
**GitHub Handle**: `devan-carlin` (GitHub doesn't allow dots in usernames)
### Day 3: Launch Day

| Time | Platform | Action |
|------|----------|--------|
| 09:00 | GitHub | Push all commits, tag release, make repo public |
| 10:00 | Medium | Publish article |
| 10:30 | Hacker News | Submit link (title: "MoE Routing Quality Paradox: Activating 64 Experts Hurt Output Quality") |
| 11:00 | X/Twitter | Post thread |
| 12:00 | HN | Engage with comments |
| 14:00 | X/Twitter | Pin thread to profile |

### Day 4: Community Day

| Time | Platform | Action |
|------|----------|--------|
| 09:00 | Reddit | Post to r/MachineLearning |
| 10:00 | Reddit | Post to r/localLLaMA |
| 11:00 | Reddit | Post to r/intel |
| 12:00 | All | Respond to comments on Reddit |
| 14:00 | Discord | Share in vLLM Discord, Intel GPU communities |
| 16:00 | All | Continue engaging |

### Day 5: Professional Day

| Time | Platform | Action |
|------|----------|--------|
| 09:00 | LinkedIn | Post professional summary |
| 10:00 | Dev.to | Cross-post Medium article |
| 11:00 | Hashnode | Cross-post Medium article |
| 12:00 | All | Respond to comments |

---

## Phase 4: Engagement & Follow-Up (Week 2+)

### 4.1 Monitor and respond
- [ ] Check HN comments daily for 3 days
- [ ] Respond to Reddit comments within 24 hours
- [ ] Engage with X/Twitter replies
- [ ] Monitor GitHub issues/PRs

### 4.2 Update based on feedback
- [ ] If people find bugs, fix and push patches
- [ ] If people ask for more data, add to GitHub
- [ ] If people suggest improvements, consider and document

### 4.3 Amplification
- [ ] Reach out to vLLM maintainers (mention the patches)
- [ ] Tag Intel GPU developers on X/Twitter
- [ ] Share in relevant Discord/Slack communities
- [ ] Consider submitting to arXiv if the quality paradox gets traction

### 4.4 Long-term
- [ ] Add to personal portfolio/website
- [ ] Reference in future projects
- [ ] Build on the findings (test other MoE models, other hardware)
- [ ] Consider follow-up articles

---

## Checklist

### Pre-Launch
- [ ] Guide committed and reviewed
- [ ] Benchmarking README created
- [ ] Results cleaned up
- [ ] GitHub repo ready for public viewing
- [ ] Medium article drafted and edited
- [ ] Charts created (4 charts)
- [ ] X/Twitter thread drafted
- [ ] LinkedIn post drafted
- [ ] Reddit posts drafted (3 variants)
- [ ] HN title tested on friends/colleagues

### Launch Day
- [ ] GitHub pushed and tagged
- [ ] Medium published
- [ ] HN submitted
- [ ] X/Twitter thread posted
- [ ] Thread pinned to profile

### Post-Launch
- [ ] Reddit posts published (3 subs)
- [ ] Discord shares made
- [ ] LinkedIn post published
- [ ] Dev.to cross-post
- [ ] Hashnode cross-post
- [ ] Comments responded to (all platforms)
- [ ] GitHub issues/PRs monitored

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| HN downvotes or negative comments | Stay calm, respond with data, don't get defensive |
| People claim the quality finding is trivial | Point to the lack of prior documentation, emphasize the empirical data |
| GitHub repo has sensitive data | Double-check before making public |
| Medium article gets low engagement | Reddit and HN will drive traffic regardless |
| People ask for reproducibility | Provide full build scripts, benchmark configs, and raw data on GitHub |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| HN upvotes | 100+ |
| Medium claps | 500+ |
| GitHub stars | 50+ |
| Reddit upvotes (total) | 200+ |
| X/Twitter impressions | 5,000+ |
| LinkedIn engagement | 50+ reactions |
| GitHub forks | 10+ |
| Comments/discussion quality | High (technical, constructive) |

---

## Next Steps

1. **Commit the guide** (can do now)
2. **Draft the Medium article** (I can help with this)
3. **Create the 4 charts** (Python/matplotlib or Figma)
4. **Draft the X/Twitter thread** (I can help with this)
5. **Review and iterate** (get feedback from friends/colleagues)
6. **Execute the launch sequence** (follow the timeline above)
