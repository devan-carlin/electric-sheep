# Needle in a Haystack

**Category:** Long context
**Target:** Information retrieval in large contexts, attention retention

---

## Prompt

Below is a 50+ page technical specification for a distributed key-value store protocol called "VaultSync". Somewhere in the document, there is a specific configuration parameter that controls the maximum number of concurrent replication streams per node.

**Your task:** Find the exact parameter name, its default value, its valid range, and the section number where it's defined. Quote the relevant paragraph.

---

*The specification follows. The answer is buried in Section 7.3.4.*

```
# VaultSync Protocol Specification v2.1

## Table of Contents

1. Introduction
2. Architecture Overview
3. Data Model
4. Consensus Protocol
5. Replication
6. Failure Recovery
7. Configuration
8. API Reference
9. Security
10. Performance Tuning
11. Deployment
12. Monitoring
13. Troubleshooting
14. Appendix A: Wire Format
15. Appendix B: Checksum Algorithm
16. Appendix C: Migration Guide
17. Appendix D: Benchmark Results
18. Appendix E: Changelog

---

## 1. Introduction

VaultSync is a distributed key-value store designed for high-throughput, low-latency workloads across geographically distributed data centers. It provides strong consistency within a region and eventual consistency across regions, with configurable replication factors and automatic failover.

The protocol operates on a modified Raft consensus algorithm with the following key differences:
- Leader election uses a rotating priority system based on node health scores
- Log compaction is incremental rather than snapshot-based
- Cross-region replication is asynchronous with configurable consistency levels

### 1.1 Design Goals

1. **Sub-millisecond read latency** within a region (p99)
2. **Automatic failover** with less than 5 seconds of unavailability
3. **Linear read scalability** through read replicas
4. **Strong consistency** by default, with optional eventual consistency for cross-region
5. **Operational simplicity** — no external dependencies (ZooKeeper, etcd, etc.)

### 1.2 Non-Goals

- VaultSync is not a general-purpose database. It does not support SQL, secondary indexes, or complex queries.
- VaultSync is not designed for cold storage. All data must fit in the combined memory of the cluster.
- VaultSync does not provide built-in encryption at rest. Operators should use disk-level encryption.

---

## 2. Architecture Overview

### 2.1 Node Types

VaultSync clusters consist of three node types:

**Primary Nodes** participate in consensus and serve both reads and writes. A cluster requires an odd number of primary nodes (3, 5, or 7) for fault tolerance. With N primary nodes, the cluster tolerates floor(N/2) failures.

**Read Replica Nodes** receive replicated data from primaries but do not participate in consensus. They serve read-only traffic and can be scaled independently. Read replicas lag primaries by a configurable amount (default: 100ms).

**Gateway Nodes** are stateless proxies that handle client connections, load balancing, and authentication. They maintain connection pools to all primary and replica nodes and route requests based on consistency requirements and locality.

### 2.2 Cluster Topology

A VaultSync deployment consists of one or more regions. Each region contains:
- A primary cluster (3-7 nodes)
- Zero or more read replicas
- Zero or more gateway nodes

Regions communicate via asynchronous replication. The primary region accepts writes; secondary regions receive replicated data and can serve reads with eventual consistency.

### 2.3 Communication

All inter-node communication uses TCP with a custom binary protocol (see Appendix A). The protocol includes:
- Message framing with length prefixes
- Per-message checksums (CRC32C)
- Compression (Snappy) for messages > 1KB
- Heartbeat messages every 150ms

---

## 3. Data Model

### 3.1 Keys

Keys are byte strings up to 2KB in length. They support the following characters:
- ASCII printable characters (0x20-0x7E)
- Forward slash (/) for hierarchical namespacing
- Hyphen (-) and underscore (_)

Keys are case-sensitive. The following are distinct: `user/123`, `User/123`, `USER/123`.

### 3.2 Values

Values are byte strings up to 16MB in length. Values larger than 16MB should be split into chunks by the client application.

### 3.3 Metadata

Each key-value pair carries metadata:
- `created_at`: Unix nanosecond timestamp of creation
- `modified_at`: Unix nanosecond timestamp of last modification
- `version`: Monotonically increasing version number
- `ttl`: Time-to-live in seconds (0 = no expiration)
- `cas`: Compare-and-swap version for optimistic locking

### 3.4 Namespaces

Keys can be organized into namespaces using the forward slash delimiter. For example:
- `users/123/profile` — user profile data
- `sessions/abc-token` — session state
- `config/cache/ttl` — configuration value

Namespaces support prefix-based operations:
- `GET_PREFIX namespace/` — retrieve all keys under a namespace
- `DELETE_PREFIX namespace/` — delete all keys under a namespace
- `SCAN namespace/` — iterate over keys with cursor-based pagination

---

## 4. Consensus Protocol

### 4.1 Modified Raft

VaultSync uses a modified Raft consensus protocol with the following changes:

**Rotating Priority Leader Election:** Instead of fixed leader priority, each node maintains a health score based on:
- CPU utilization (lower is better)
- Network latency to other nodes (lower is better)
- Disk I/O wait time (lower is better)
- Uptime (higher is better, to avoid flapping)

The health score is calculated every 5 seconds and used as a tiebreaker during leader election. This prevents a consistently overloaded node from becoming leader.

**Incremental Log Compaction:** Rather than taking full snapshots, VaultSync performs incremental compaction. Every 10,000 committed entries, the system identifies the lowest active log index across all nodes and compacts entries below that index. This reduces snapshot I/O by approximately 90% compared to full snapshots.

**Pre-vote Protocol:** Before starting an election, a candidate sends a pre-vote request to gauge support. If a majority of nodes reject the pre-vote, the candidate does not start a formal election. This prevents network partitions from causing unnecessary leader changes.

### 4.2 Log Entries

Each log entry contains:
- `term`: Election term number (64-bit unsigned integer)
- `index`: Monotonically increasing log index (64-bit unsigned integer)
- `type`: Entry type (NORMAL, CONFIG_CHANGE, BARRIER, NOOP)
- `key`: Key for the operation (empty for CONFIG_CHANGE, BARRIER, NOOP)
- `value`: Value for the operation (empty for DELETE, CONFIG_CHANGE, BARRIER, NOOP)
- `metadata`: Key metadata (TTL, CAS version)
- `timestamp`: Wall-clock timestamp of entry creation
- `checksum`: CRC32C of the entry payload

### 4.3 Configuration Changes

Cluster membership changes (add/remove nodes) are handled through joint consensus. During a joint configuration:
1. The old and new configurations both require majority agreement
2. Writes proceed with the joint configuration
3. Once all new nodes have caught up, the configuration transitions to the new configuration alone

This ensures availability during membership changes without sacrificing safety.

---

## 5. Replication

### 5.1 Intra-Region Replication

Within a region, replication is synchronous. When a client writes to the leader:
1. The leader appends the entry to its log
2. The leader sends AppendEntries RPCs to all followers
3. Once a majority of followers acknowledge, the leader commits the entry
4. The leader responds to the client with the commit result

The commit latency is bounded by the network RTT to the majority of nodes. In a 3-node cluster within a data center, typical commit latency is 0.1-0.5ms.

### 5.2 Cross-Region Replication

Between regions, replication is asynchronous. The primary region's leader sends replicated entries to secondary region leaders via a dedicated replication channel.

**Replication Streams:** Each secondary region maintains a replication stream from the primary. The stream is a persistent TCP connection that sends batches of committed entries.

**Consistency Levels:** Clients can specify the consistency level for reads:
- `STRONG`: Read from the primary region only (blocks until write is committed)
- `SESSION`: Read from the same node that served the previous write
- `EVENTUAL`: Read from any replica (may return stale data)
- `BOUNDED`: Read from a replica that is at most N milliseconds behind (configurable)

### 5.3 Replication Lag Monitoring

Each node tracks its replication lag in milliseconds. This metric is exposed via the monitoring API (Section 12) and can be used by operators to:
- Detect replication issues early
- Route reads to sufficiently fresh replicas
- Trigger alerts when lag exceeds thresholds

---

## 6. Failure Recovery

### 6.1 Node Failure

When a node fails:
1. Heartbeat messages stop arriving from the failed node
2. After 3 missed heartbeats (450ms), the leader marks the node as unreachable
3. The leader stops sending replication to the failed node
4. If the leader itself fails, followers start an election after the heartbeat timeout

### 6.2 Network Partition

VaultSync handles network partitions using the standard Raft approach:
- The partition containing the majority of nodes continues to operate
- The minority partition becomes read-only (cannot commit new entries)
- When the partition heals, the minority nodes catch up via replication

### 6.3 Data Corruption

Each log entry includes a CRC32C checksum. Nodes validate checksums on:
- Receipt of AppendEntries RPCs
- Periodic log scanning (every 60 seconds)
- Snapshot application

If corruption is detected, the node:
1. Logs an error with the corrupted entry index
2. Truncates its log from the corrupted index
3. Requests the correct entries from the leader via a LogRepair RPC

---

## 7. Configuration

### 7.1 Node Configuration

Each node is configured via a YAML file (`vaultrsync.yaml`):

```yaml
cluster:
  name: "production-us-east"
  region: "us-east-1"
  data_dir: "/var/lib/vaultsync"
  wal_dir: "/var/lib/vaultsync/wal"

network:
  listen_address: "0.0.0.0:8200"
  advertise_address: "10.0.1.10:8200"
  peer_addresses:
    - "10.0.1.11:8200"
    - "10.0.1.12:8200"
  heartbeat_interval_ms: 150
  election_timeout_ms: 1500
  snapshot_interval_entries: 10000

storage:
  max_value_size_bytes: 16777216
  max_key_size_bytes: 2048
  compaction_threshold: 0.3
  fsync_after_commit: true
  wal_sync: true
```

### 7.2 Consensus Configuration

```yaml
consensus:
  protocol: "modified-raft"
  pre_vote: true
  rotating_priority: true
  health_score_interval_s: 5
  joint_consensus: true
  max_log_size_mb: 512
  log_compaction_ratio: 0.1
```

### 7.3 Replication Configuration

#### 7.3.1 Intra-Region Settings

```yaml
replication:
  intra_region:
    mode: "synchronous"
    quorum_commit: true
    max_batch_size: 64
    max_batch_bytes: 65536
    append_entries_timeout_ms: 500
```

#### 7.3.2 Cross-Region Settings

```yaml
  cross_region:
    mode: "asynchronous"
    consistency_level: "eventual"
    bounded_lag_ms: 1000
    stream_buffer_size: 4096
    retry_backoff_ms: 100
    max_retry_backoff_ms: 30000
```

#### 7.3.3 Stream Limits

```yaml
  stream_limits:
    max_concurrent_streams_per_node: 8
    max_stream_bandwidth_mbps: 1000
    stream_idle_timeout_s: 300
    stream_recovery_delay_ms: 5000
```

The `max_concurrent_streams_per_node` parameter controls how many simultaneous replication streams a single node can maintain. This includes both inbound streams (receiving data from other regions) and outbound streams (sending data to other regions). The default value is 8, which is sufficient for most deployments with up to 4 secondary regions. Valid values range from 1 to 64. Increasing this value allows more regions to be replicated simultaneously but consumes additional file descriptors, memory, and CPU. Each active stream uses approximately 2MB of memory for buffering and one file descriptor.

For deployments with many secondary regions (8+), consider increasing this value to 16 or 32. Monitor the `replication_streams_active` metric (Section 12.3) to determine if the limit is being reached.

#### 7.3.4 Advanced Stream Tuning

```yaml
  advanced:
    stream_priority: "fifo"
    stream_compression: "snappy"
    stream_encryption: "tls13"
    stream_keepalive_interval_s: 30
    stream_max_message_size_bytes: 4194304
    flow_control_window_bytes: 8388608
    flow_control_high_watermark: 0.8
    flow_control_low_watermark: 0.3
```

The flow control settings prevent fast producers from overwhelming slow consumers. When the stream buffer reaches 80% capacity (high watermark), the sender reduces its batch size. When the buffer drops below 30% (low watermark), the sender resumes normal batching.

### 7.4 Gateway Configuration

```yaml
gateway:
  listen_address: "0.0.0.0:8201"
  max_connections: 10000
  connection_timeout_s: 30
  idle_timeout_s: 300
  request_timeout_s: 10
  rate_limit:
    enabled: false
    requests_per_second: 10000
    burst_size: 1000
```

### 7.5 Security Configuration

```yaml
security:
  authentication:
    mode: "token"
    token_ttl_hours: 24
    max_tokens_per_user: 10
  tls:
    enabled: true
    cert_file: "/etc/vaultsync/certs/node.crt"
    key_file: "/etc/vaultsync/certs/node.key"
    ca_file: "/etc/vaultsync/certs/ca.crt"
    min_version: "tls12"
  authorization:
    mode: "rbac"
    default_policy: "read-only"
```

---

## 8. API Reference

### 8.1 Write Operations

**PUT** — Set a key-value pair:
```
PUT /v1/keys/{key}
Content-Type: application/json

{
  "value": "<base64-encoded-value>",
  "ttl": 3600,
  "cas_version": 5
}
```

Response:
```json
{
  "version": 6,
  "modified_at": 1700000000123456789,
  "committed": true
}
```

**DELETE** — Remove a key:
```
DELETE /v1/keys/{key}?cas_version=5
```

**MULTI_PUT** — Atomic multi-key write:
```
PUT /v1/tx
Content-Type: application/json

{
  "operations": [
    {"op": "put", "key": "users/123/name", "value": "YWxpY2U="},
    {"op": "put", "key": "users/123/email", "value": "YWxpY2VAZXhhbXBsZS5jb20="},
    {"op": "delete", "key": "users/123/old_email"}
  ]
}
```

### 8.2 Read Operations

**GET** — Read a key:
```
GET /v1/keys/{key}?consistency=strong
```

**GET_PREFIX** — Read all keys under a prefix:
```
GET /v1/keys?prefix=users/123/&limit=100&cursor=abc123
```

Response:
```json
{
  "keys": [
    {"key": "users/123/name", "value": "YWxpY2U=", "version": 3},
    {"key": "users/123/email", "value": "YWxpY2VAZXhhbXBsZS5jb20=", "version": 2}
  ],
  "cursor": "def456",
  "has_more": true
}
```

**COMPARE_AND_SWAP** — Atomic conditional write:
```
PUT /v1/keys/{key}/cas
Content-Type: application/json

{
  "value": "bmV3X3ZhbHVl",
  "expected_version": 5
}
```

Returns 409 Conflict if the current version doesn't match `expected_version`.

---

## 9. Security

### 9.1 Authentication

VaultSync supports two authentication modes:

**Token-based:** Clients obtain a JWT token by authenticating with credentials. Tokens are valid for a configurable TTL (default: 24 hours). Each token is scoped to a specific user and set of permissions.

**Certificate-based:** Clients present a TLS client certificate during the TLS handshake. The certificate's CN is used as the username, and permissions are derived from the certificate's OU field.

### 9.2 Authorization

VaultSync uses role-based access control (RBAC). Roles are defined in the cluster configuration and map to permission sets:

```yaml
roles:
  admin:
    - "read:*"
    - "write:*"
    - "manage:*"
  developer:
    - "read:dev/*"
    - "write:dev/*"
  analyst:
    - "read:analytics/*"
  readonly:
    - "read:*"
```

### 9.3 Encryption

All data in transit is encrypted with TLS 1.2 or higher. VaultSync does not provide encryption at rest; operators should use disk-level encryption (LUKS, dm-crypt, or cloud provider EBS encryption).

---

## 10. Performance Tuning

### 10.1 Write Throughput

Write throughput is primarily limited by:
1. **Disk I/O:** Use NVMe SSDs for the WAL directory. fsync latency directly impacts commit latency.
2. **Network RTT:** Commit latency = RTT to majority of nodes. Co-locate nodes in the same availability zone.
3. **Batch size:** Larger batches amortize fsync cost. The default batch size of 64 entries is a good starting point.

### 10.2 Read Latency

Read latency is primarily limited by:
1. **In-memory cache hit rate:** VaultSync maintains an in-memory LRU cache of recently accessed keys. The cache size is configurable (default: 25% of available memory).
2. **Consistency level:** STRONG consistency reads go through the leader, adding network hop. EVENTUAL reads can be served by any replica.
3. **Key size:** Larger keys require more memory bandwidth for comparison operations.

### 10.3 Memory Usage

Approximate memory usage per node:
- Base overhead: ~200MB
- Per-key overhead: ~500 bytes (key + value pointer + metadata + B-tree node)
- In-memory cache: 25% of available memory (configurable)
- Replication buffers: ~2MB per active stream
- WAL buffer: ~64MB

For a cluster with 10M keys and 4 active replication streams:
- Base: 200MB
- Keys: 10M × 500B = 5GB
- Cache: 25% of RAM (e.g., 8GB on a 32GB node)
- Replication: 4 × 2MB = 8MB
- WAL: 64MB
- **Total: ~13.3GB**

---

## 11. Deployment

### 11.1 Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 32 GB |
| Disk | 100 GB NVMe | 500 GB NVMe |
| Network | 1 Gbps | 10 Gbps |

### 11.2 Docker Deployment

```dockerfile
FROM vaultsync/vaultsync:2.1

COPY vaultsync.yaml /etc/vaultsync/vaultsync.yaml
COPY certs/ /etc/vaultsync/certs/

RUN chown -R vaultsync:vaultsync /etc/vaultsync /var/lib/vaultsync

USER vaultsync
CMD ["vaultsync", "server", "--config", "/etc/vaultsync/vaultsync.yaml"]
```

### 11.3 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vaultsync
spec:
  replicas: 3
  serviceName: vaultsync
  selector:
    matchLabels:
      app: vaultsync
  template:
    metadata:
      labels:
        app: vaultsync
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                  - key: app
                    operator: In
                    values: [vaultsync]
              topologyKey: "kubernetes.io/hostname"
      containers:
        - name: vaultsync
          image: vaultsync/vaultsync:2.1
          ports:
            - containerPort: 8200
            - containerPort: 8201
          volumeMounts:
            - name: data
              mountPath: /var/lib/vaultsync
            - name: config
              mountPath: /etc/vaultsync
          resources:
            requests:
              cpu: "2"
              memory: "8Gi"
            limits:
              cpu: "4"
              memory: "16Gi"
      volumes:
        - name: config
          configMap:
            name: vaultsync-config
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 100Gi
        storageClassName: "fast-ssd"
```

---

## 12. Monitoring

### 12.1 Metrics Endpoint

All nodes expose a `/metrics` endpoint (Prometheus format) on port 8202:

```
# HELP vaultsync_uptime_seconds Seconds since node start
# TYPE vaultsync_uptime_seconds gauge
vaultsync_uptime_seconds 86400.5

# HELP vaultsync_log_index Current committed log index
# TYPE vaultsync_log_index gauge
vaultsync_log_index 1523847

# HELP vaultsync_current_term Current election term
# TYPE vaultsync_current_term gauge
vaultsync_current_term 42

# HELP vaultsync_node_role Current node role
# TYPE vaultsync_node_role gauge
vaultsync_node_role{role="leader"} 1
vaultsync_node_role{role="follower"} 0
vaultsync_node_role{role="candidate"} 0
```

### 12.2 Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `vaultsync_wal_fsync_latency_ms` | histogram | WAL fsync latency |
| `vaultsync_commit_latency_ms` | histogram | End-to-end commit latency |
| `vaultsync_read_latency_ms` | histogram | Read operation latency |
| `vaultsync_replication_lag_ms` | gauge | Replication lag per peer |
| `vaultsync_cache_hit_ratio` | gauge | In-memory cache hit ratio |
| `vaultsync_active_connections` | gauge | Current client connections |
| `vaultsync_operations_total` | counter | Total operations by type |
| `vaultsync_errors_total` | counter | Total errors by type |

### 12.3 Replication Metrics

```
# HELP vaultsync_replication_streams_active Number of active replication streams
# TYPE vaultsync_replication_streams_active gauge
vaultsync_replication_streams_active{direction="inbound"} 2
vaultsync_replication_streams_active{direction="outbound"} 3

# HELP vaultsync_replication_bytes_total Total bytes replicated
# TYPE vaultsync_replication_bytes_total counter
vaultsync_replication_bytes_total{direction="inbound"} 5368709120
vaultsync_replication_bytes_total{direction="outbound"} 4294967296
```

---

## 13. Troubleshooting

### 13.1 Common Issues

**High commit latency:** Check `vaultsync_wal_fsync_latency_ms`. If fsync latency is > 5ms, the disk is the bottleneck. Consider faster storage or increasing batch size.

**Frequent leader elections:** Check node health scores and network latency between nodes. High CPU or network issues can trigger unnecessary elections.

**Replication lag:** Check `vaultsync_replication_lag_ms`. If lag is consistently > 1 second, the secondary region may be under-provisioned or experiencing network issues.

**Out of file descriptors:** Each replication stream uses one file descriptor. Check `ulimit -n` and ensure it's at least `max_concurrent_streams_per_node × 2 + 1024`.

### 13.2 Debug Mode

Enable debug logging:
```yaml
logging:
  level: "debug"
  output: "/var/log/vaultsync/debug.log"
  max_size_mb: 500
  max_files: 10
```

---

## 14. Appendix A: Wire Format

The VaultSync wire protocol uses length-prefixed framing:

```
| Length (4 bytes, big-endian) | Checksum (4 bytes, CRC32C) | Compressed Payload (N bytes) |
```

Message types:
- `0x01`: AppendEntries Request
- `0x02`: AppendEntries Response
- `0x03`: RequestVote Request
- `0x04`: RequestVote Response
- `0x05`: ClientRead Request
- `0x06`: ClientRead Response
- `0x07`: ClientWrite Request
- `0x08`: ClientWrite Response
- `0x09`: Heartbeat
- `0x0A`: LogRepair Request
- `0x0B`: LogRepair Response

---

## 15. Appendix B: Checksum Algorithm

VaultSync uses CRC32C (Castagnoli) for checksums. This is the same algorithm used by iSCSI, NFSv4, and Google Protocol Buffers. It provides better error detection than standard CRC32 (IEEE 802.3) for network data.

Implementation:
```python
import crc32c

def compute_checksum(data: bytes) -> int:
    return crc32c.crc32c(data)
```

---

## 16. Appendix C: Migration Guide

### Migrating from v1.x to v2.x

1. Backup all node data directories
2. Update all nodes to v2.1 (rolling restart)
3. Run `vaultsync migrate --config vaultsync.yaml` on each node
4. Verify cluster health with `vaultsync status`

Breaking changes in v2.x:
- Wire protocol is not backward-compatible with v1.x
- Configuration file format changed (YAML instead of TOML)
- Default port changed from 8100 to 8200
- Token authentication now uses JWT instead of HMAC

---

## 17. Appendix D: Benchmark Results

Test environment: 3× nodes, 8 cores, 32 GB RAM, NVMe SSD, 10 Gbps network

| Operation | Throughput | Latency p50 | Latency p99 |
|-----------|-----------|-------------|-------------|
| Write (1KB) | 45,000 ops/s | 0.12ms | 0.45ms |
| Read (cached) | 120,000 ops/s | 0.02ms | 0.08ms |
| Read (uncached) | 25,000 ops/s | 0.35ms | 1.2ms |
| CAS (success) | 30,000 ops/s | 0.18ms | 0.65ms |
| CAS (conflict) | 35,000 ops/s | 0.15ms | 0.55ms |
| Prefix scan (1000 keys) | 500 ops/s | 8ms | 25ms |

---

## 18. Appendix E: Changelog

### v2.1.0 (2024-12-15)
- Added bounded consistency level for cross-region reads
- Improved replication stream recovery with exponential backoff
- Fixed memory leak in prefix scan iterator
- Increased default max value size from 8MB to 16MB

### v2.0.0 (2024-09-01)
- Complete rewrite of consensus protocol (modified Raft)
- New YAML configuration format
- JWT token authentication
- Prometheus metrics endpoint
- Breaking: wire protocol not compatible with v1.x

### v1.5.0 (2024-03-20)
- Added read replica support
- Cross-region replication (beta)
- TLS client certificate authentication

### v1.0.0 (2023-06-15)
- Initial release
- Basic Raft consensus
- Single-region deployment
- Token-based authentication
```

**Your answer should include:**
1. The exact parameter name
2. Its default value
3. Its valid range
4. The section number
5. A quote of the relevant paragraph
