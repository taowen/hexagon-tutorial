/*
 * ch05: ARM ↔ DSP 共享消息协议
 *
 * 对应 llama.cpp 的 htp-msg.h。
 * llama.cpp 定义了 29 种 op code（HTP_OP_MUL_MAT, HTP_OP_ADD, ...）
 * 我们简化为 2 种，展示同样的 dispatch 模式。
 */

#ifndef DSPQUEUE_DEMO_SHARED_H
#define DSPQUEUE_DEMO_SHARED_H

#include <stdint.h>

/*
 * Op codes — 对应 llama.cpp 的 HTP_OP_xxx
 *
 * llama.cpp 有:
 *   HTP_OP_MUL_MAT       = 0   (HMX 矩阵乘)
 *   HTP_OP_MUL            = 1   (逐元素乘)
 *   HTP_OP_ADD            = 2   (逐元素加)
 *   HTP_OP_RMS_NORM       = 5   (RMSNorm)
 *   HTP_OP_SCALE          = 6   (标量缩放)
 *   ... 总共 29 种
 *
 * 我们简化为 2 种:
 */
#define OP_SCALE    1    /* out[i] = in[i] * factor  — 类似 HTP_OP_SCALE */
#define OP_ADD      2    /* out[i] = a[i] + b[i]    — 类似 HTP_OP_ADD   */
#define OP_ECHO     99   /* Echo（测试连通性）                             */

/*
 * 请求消息 — 对应 llama.cpp 的 htp_general_req
 *
 * llama.cpp 的 htp_general_req 有 ~200 字节，包含：
 *   - uint32_t op           (op code)
 *   - int32_t  op_params[16] (算子参数)
 *   - htp_tensor src0, src1, dst (张量元数据)
 *
 * 我们简化为 16 字节:
 */
struct demo_req {
    uint32_t op;              /* OP_SCALE, OP_ADD, OP_ECHO */
    uint32_t param;           /* OP_SCALE: scale factor; 其他: early_wakeup_limit */
    uint32_t n_elem;          /* 元素数量（uint8_t 单位） */
    uint32_t reserved;
};

/*
 * 响应消息 — 对应 llama.cpp 的 htp_general_rsp
 *
 * llama.cpp 的 htp_general_rsp 有：
 *   - uint32_t op
 *   - uint32_t status       (HTP_STATUS_OK / ERR)
 *   - uint32_t prof_usecs   (执行耗时)
 *   - uint32_t prof_cycles
 */
struct demo_rsp {
    uint32_t op;
    uint32_t status;          /* 0 = OK */
};

/* 队列参数 */
#define DEMO_MAX_MESSAGE_SIZE  sizeof(struct demo_req)
#define DEMO_MAX_BUFFERS       3    /* src0(input_a), src1(input_b), dst(output) */

#endif /* DSPQUEUE_DEMO_SHARED_H */
