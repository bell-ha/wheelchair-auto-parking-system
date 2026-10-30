/*
 * uwb_anchor — 앵커 (차에 다는 쪽). 자기 번호로 온 poll에만 답한다 (SS-TWR 응답자).
 * 거리 계산은 태그가 한다. 앵커는 전원만 있으면 되고 PC 연결은 필요 없다.
 * 태그가 바퀴마다 방송하는 결과(report)를 받아 태그와 같은 R/E 줄로 출력한다
 * → 앵커 하나(A1)만 컴퓨터에 꽂아도 태그의 거리 전부를 볼 수 있다.
 *
 * 측량: 컴퓨터가 시리얼로 "SURVEY"를 보내면 이 앵커가 앵커끼리의 거리를 잰다.
 *   1) 자기가 다른 앵커 모두와 잰다 (쌍마다 SURVEY_N번, 중앙값)
 *   2) 다른 앵커에게 "너보다 번호가 큰 앵커들과 재서 알려줘"라고 시키고 결과를 받는다
 *   → 앵커가 3개면 A1-A2, A1-A3, A2-A3 세 거리가 모두 나온다. 화면이 이걸로 앵커 위치를 계산한다.
 *   측량하는 2~3초 동안은 태그 측정이 몇 번 실패한다.
 *
 * 업로드 전에 아래 "보드마다 바꾸는 값" 두 줄만 확인한다.
 *
 * 출처: Makerfabs-ESP32-UWB-DW3000 example/range/range_tx (Qorvo ex_06b 기반)
 * 바꾼 것:
 *   - 번호(MY_ID): 자기 번호로 온 poll에만 답하고, 물어본 쪽에게 답한다 → 앵커 여러 개를 같이 켤 수 있다
 *   - 부팅 시 번호·MAC·안테나 지연을 '#' 줄로 출력 (보드 확인용)
 *   - 1초마다 태그별 응답 횟수 출력
 *   - 태그의 결과 방송 출력, 앵커끼리 측량
 *
 * 무선 메시지 — IEEE 802.15.4 데이터 프레임 모양. 주소 2바이트에 번호 글자 2개("A1", "T2")를 넣는다
 *   poll        41 88 seq CA DE | 받는쪽 2 | 보내는쪽 2 | E0                                   (+CRC 2 = 12바이트)
 *   response    41 88 seq CA DE | 받는쪽 2 | 보내는쪽 2 | E1 | poll 받은 시각 4 | 응답 보낸 시각 4 (+CRC 2 = 20바이트)
 *   report      41 88 seq CA DE | FF FF | 태그 2 | E2 | 바퀴 번호 2 | 개수 1 | 앵커마다 28바이트   (구조는 uwb_tag.ino 머리말)
 *   survey_cmd  41 88 seq CA DE | 받는쪽 2 | 지휘 앵커 2 | E3                                 "너보다 큰 번호들과 재라"
 *   survey_res  41 88 seq CA DE | FF FF | 잰 앵커 2 | E4 | 개수 1 | 칸마다: 상대 번호 2 | 거리 mm 4 | 성공 1 | 시도 1
 *
 * 출력 (115200bps):
 *   # uwb_anchor id=A1 mac=.. ant_dly=.. dev_id=..    부팅 정보
 *   # replies=<합계> per 1s (T1=.. T2=..)             지난 1초 동안 태그에게 응답한 횟수 (요청이 있을 때만)
 *   R,ms,seq,tag,anchor,dist_m,clk_ppm,C,F1,F2,F3,N,D 태그가 방송한 결과 (seq = 바퀴 번호, ms = 이 앵커가 받은 시각)
 *   E,ms,seq,tag,anchor,reason
 *   # survey start / # survey done                    측량 시작·끝
 *   S,ms,from,to,dist_m,n_ok,n_try                    앵커끼리 거리 (중앙값. 전부 실패면 dist_m = -1)
 */
#include "dw3000.h"
#include "SPI.h"
#include <esp_system.h>

extern SPISettings _fastSPI;

// ======================================================
// 보드마다 바꾸는 값 — 업로드 전에 확인
// ======================================================
// 터미널에서는 코드를 안 고치고 번호를 넘길 수 있다: --build-property 'compiler.cpp.extra_flags=-DMY_ID="A2"' (docs/UWB_설명.txt 5절)
#ifndef MY_ID
#define MY_ID "A1"        // 앵커 번호 "A1", "A2", "A3". 두 보드에 같은 번호를 넣으면 안 된다
#endif
#ifndef ANT_DLY
#define ANT_DLY 16385     // 이 보드의 안테나 지연. 보정 전 기본값 16385 (보정값은 docs/UWB_설명.txt 2절 보드 목록)
#endif

// 측량할 때 서로 재는 앵커 번호 (태그 코드의 ANCHORS와 같게)
static const char *ANCHORS[] = {"A1", "A2", "A3"};

// ======================================================
// Pin map — 보드 내부 배선 (ESP32 <-> DW3000). 바꾸지 말 것
// ======================================================
#define PIN_RST 27
#define PIN_IRQ 34
#define PIN_SS 4

// ======================================================
// Ranging — 태그 코드와 반드시 같은 값이어야 하는 것들
// ======================================================
#define POLL_RX_TO_RESP_TX_DLY_UUS 900   // poll 받고 응답 보내기까지 (묻는 쪽의 대기시간과 짝)
#define POLL_TX_TO_RESP_RX_DLY_UUS 600   // 측량 때 이 앵커가 묻는 쪽이 되면: poll 보내고 이만큼 쉬었다가 수신
#define RESP_RX_TIMEOUT_UUS 400          //   그 뒤 이 시간 안에 응답이 없으면 실패

#define SURVEY_N 20                      // 측량: 쌍마다 재는 횟수 (중앙값을 쓴다)
#define SURVEY_GAP_MS 5                  //   재는 사이 쉬는 시간
#define SURVEY_WAIT_MS 3000              //   다른 앵커의 측량 결과를 기다리는 시간

#define FUNC_POLL 0xE0
#define FUNC_RESP 0xE1
#define FUNC_REPORT 0xE2
#define FUNC_SURVEY_CMD 0xE3
#define FUNC_SURVEY_RES 0xE4
#define REPORT_HDR 13      // 머리 10 + 바퀴 번호 2 + 개수 1
#define REPORT_ENTRY 28
#define SURVEY_HDR 11      // 머리 10 + 개수 1
#define SURVEY_ENTRY 8
#define MAX_FRAME 127      // 표준 프레임 최대 (CRC 포함)
#define IDX_SEQ 2
#define IDX_DST 5          // 받는 쪽 번호 2바이트
#define IDX_SRC 7          // 보내는 쪽 번호 2바이트
#define IDX_FUNC 9
#define IDX_POLL_RX_TS 10
#define IDX_RESP_TX_TS 14
#define POLL_LEN 12        // CRC 포함
#define RESP_LEN 20

/* 채널 5, 프리앰블 128, 코드 9(PRF 64 MHz), 6.8 Mbps. STS·PDoA 끔 (단일 안테나 칩) */
static dwt_config_t config = {
    5,                /* Channel number. */
    DWT_PLEN_128,     /* Preamble length. Used in TX only. */
    DWT_PAC8,         /* Preamble acquisition chunk size. Used in RX only. */
    9,                /* TX preamble code. Used in TX only. */
    9,                /* RX preamble code. Used in RX only. */
    1,                /* non-standard 8 symbol SFD */
    DWT_BR_6M8,       /* Data rate. */
    DWT_PHRMODE_STD,  /* PHY header mode. */
    DWT_PHRRATE_STD,  /* PHY header rate. */
    (129 + 8 - 8),    /* SFD timeout (preamble length + 1 + SFD length - PAC size). Used in RX only. */
    DWT_STS_MODE_OFF, /* STS disabled */
    DWT_STS_LEN_64,   /* STS length */
    DWT_PDOA_M0       /* PDOA mode off */
};

extern dwt_txconfig_t txconfig_options;

static const int N_ANCHORS = sizeof(ANCHORS) / sizeof(ANCHORS[0]);
static uint8_t frame_seq_nb = 0;
static uint8_t rx_buffer[MAX_FRAME];

// 태그별 응답 횟수 (1초마다 출력하고 비운다). 처음 보는 태그 번호를 차례로 칸에 넣는다
#define MAX_TAGS 4
static char tag_ids[MAX_TAGS][3];
static uint32_t tag_replies[MAX_TAGS];
static int n_tags = 0;
static uint32_t replies_total = 0;
static uint32_t last_report_ms = 0;

// 측량 결과 한 칸
struct Pair
{
  const char *to;
  int32_t mm;        // 중앙값. 전부 실패면 -1
  uint8_t ok, tries;
};

static void halt(const char *msg)
{
  Serial.printf("# FATAL %s\n", msg);
  while (1)
    ;
}

static bool same_id(const uint8_t *field, const char *id)
{
  return field[0] == (uint8_t)id[0] && field[1] == (uint8_t)id[1];
}

static bool is_me(const uint8_t *field) { return same_id(field, MY_ID); }

static uint16_t get16(const uint8_t *p) { return p[0] | (p[1] << 8); }
static uint32_t get32(const uint8_t *p) { return p[0] | (p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static void put32(uint8_t *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }

static void count_reply(const uint8_t *src)
{
  if (src[0] == 'A')
    return;   // 측량 중 다른 앵커에게 한 응답은 세지 않는다
  replies_total++;
  for (int i = 0; i < n_tags; i++)
    if (tag_ids[i][0] == src[0] && tag_ids[i][1] == src[1])
    {
      tag_replies[i]++;
      return;
    }
  if (n_tags < MAX_TAGS)
  {
    tag_ids[n_tags][0] = src[0];
    tag_ids[n_tags][1] = src[1];
    tag_ids[n_tags][2] = 0;
    tag_replies[n_tags++] = 1;
  }
}

static void report()
{
  uint32_t now = millis();
  if (now - last_report_ms < 1000)
    return;
  if (replies_total > 0)
  {
    Serial.printf("# replies=%lu per 1s (", (unsigned long)replies_total);
    for (int i = 0; i < n_tags; i++)
      Serial.printf("%s%s=%lu", i ? " " : "", tag_ids[i], (unsigned long)tag_replies[i]);
    Serial.println(")");
  }
  replies_total = 0;
  for (int i = 0; i < n_tags; i++)
    tag_replies[i] = 0;
  last_report_ms = now;
}

/* 프레임을 하나 보낸다 (답을 기다리지 않음). CRC 2바이트는 칩이 붙인다 */
static void send_frame(uint8_t *msg, uint16_t len)
{
  dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS_BIT_MASK);
  dwt_writetxdata(len, msg, 0);
  dwt_writetxfctrl(len + 2, 0, 0);
  dwt_starttx(DWT_START_TX_IMMEDIATE);
  while (!(dwt_read32bitreg(SYS_STATUS_ID) & SYS_STATUS_TXFRS_BIT_MASK))
  {
  };
  dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS_BIT_MASK);
}

static void fill_header(uint8_t *msg, const uint8_t *dst, uint8_t func)
{
  msg[0] = 0x41; msg[1] = 0x88; msg[IDX_SEQ] = frame_seq_nb++; msg[3] = 0xCA; msg[4] = 0xDE;
  msg[IDX_DST] = dst[0]; msg[IDX_DST + 1] = dst[1];
  msg[IDX_SRC] = MY_ID[0]; msg[IDX_SRC + 1] = MY_ID[1];
  msg[IDX_FUNC] = func;
}

// ======================================================
// 응답 (태그든 측량하는 앵커든, 내 번호로 온 poll에 답한다)
// ======================================================
static void respond_to_poll()
{
  uint8_t tx_resp[RESP_LEN - 2];
  uint64_t poll_rx_ts = get_rx_timestamp_u64();

  /* 응답은 poll 수신 시각 + 고정 지연에 예약 송신. 그래야 송신 시각을 미리 알고 메시지에 담을 수 있다 */
  uint32_t resp_tx_time = (poll_rx_ts + (POLL_RX_TO_RESP_TX_DLY_UUS * UUS_TO_DWT_TIME)) >> 8;
  dwt_setdelayedtrxtime(resp_tx_time);
  /* 실제 송신 타임스탬프 = 예약 시각 + 안테나 지연 */
  uint64_t resp_tx_ts = (((uint64_t)(resp_tx_time & 0xFFFFFFFEUL)) << 8) + ANT_DLY;

  fill_header(tx_resp, &rx_buffer[IDX_SRC], FUNC_RESP);   // 물어본 쪽에게
  resp_msg_set_ts(&tx_resp[IDX_POLL_RX_TS], poll_rx_ts);
  resp_msg_set_ts(&tx_resp[IDX_RESP_TX_TS], resp_tx_ts);

  dwt_writetxdata(sizeof(tx_resp), tx_resp, 0);
  dwt_writetxfctrl(sizeof(tx_resp) + 2, 0, 1);

  /* 예약 시각을 이미 지나쳤으면 실패 — 이번 교환은 버리고 다음 poll을 기다린다 */
  if (dwt_starttx(DWT_START_TX_DELAYED) == DWT_SUCCESS)
  {
    while (!(dwt_read32bitreg(SYS_STATUS_ID) & SYS_STATUS_TXFRS_BIT_MASK))
    {
    };
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS_BIT_MASK);
    count_reply(&rx_buffer[IDX_SRC]);
  }
}

/* 태그가 방송한 결과를 태그와 같은 R/E 줄로 출력 (live_view.py가 그대로 읽는다) */
static void print_tag_report(uint32_t frame_len)
{
  int n = rx_buffer[12];
  if (REPORT_HDR + n * REPORT_ENTRY + 2 > (int)frame_len)
    return;   // 잘린 메시지
  char tag[3] = {(char)rx_buffer[IDX_SRC], (char)rx_buffer[IDX_SRC + 1], 0};
  unsigned cyc = get16(&rx_buffer[10]);
  unsigned long ms = millis();
  for (int i = 0; i < n; i++)
  {
    const uint8_t *p = &rx_buffer[REPORT_HDR + i * REPORT_ENTRY];
    char anchor[3] = {(char)p[0], (char)p[1], 0};
    if (p[2] == 0)
      Serial.printf("R,%lu,%u,%s,%s,%.3f,%.2f,%lu,%lu,%lu,%lu,%u,%u\n", ms, cyc, tag, anchor,
                    (int32_t)get32(&p[3]) / 1000.0, (int16_t)get16(&p[7]) / 100.0,
                    (unsigned long)get32(&p[9]), (unsigned long)get32(&p[13]),
                    (unsigned long)get32(&p[17]), (unsigned long)get32(&p[21]),
                    (unsigned)get16(&p[25]), (unsigned)p[27]);
    else
      Serial.printf("E,%lu,%u,%s,%s,%s\n", ms, cyc, tag, anchor,
                    p[2] == 1 ? "timeout" : p[2] == 2 ? "rx_err" : "bad_frame");
  }
}

// ======================================================
// 측량 — 이 앵커가 묻는 쪽이 되어 다른 앵커와 잰다
// ======================================================

/* 상대 앵커에게 한 번 묻고 거리(m)를 구한다. 계산은 태그 코드와 같다 */
static bool range_to(const char *target, double *dist)
{
  uint8_t tx_poll[POLL_LEN - 2];
  fill_header(tx_poll, (const uint8_t *)target, FUNC_POLL);
  dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS_BIT_MASK);
  dwt_writetxdata(sizeof(tx_poll), tx_poll, 0);
  dwt_writetxfctrl(sizeof(tx_poll) + 2, 0, 1);
  dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);

  uint32_t status_reg;
  while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG_BIT_MASK | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR)))
  {
  };
  if (!(status_reg & SYS_STATUS_RXFCG_BIT_MASK))
  {
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
    return false;
  }
  dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG_BIT_MASK);
  uint32_t frame_len = dwt_read32bitreg(RX_FINFO_ID) & RXFLEN_MASK;
  if (frame_len != RESP_LEN)
    return false;
  dwt_readrxdata(rx_buffer, frame_len, 0);
  if (!(rx_buffer[IDX_FUNC] == FUNC_RESP && is_me(&rx_buffer[IDX_DST]) && same_id(&rx_buffer[IDX_SRC], target)))
    return false;

  uint32_t poll_tx_ts = dwt_readtxtimestamplo32();
  uint32_t resp_rx_ts = dwt_readrxtimestamplo32();
  float clockOffsetRatio = ((float)dwt_readclockoffset()) / (uint32_t)(1 << 26);
  uint32_t poll_rx_ts, resp_tx_ts;
  resp_msg_get_ts(&rx_buffer[IDX_POLL_RX_TS], &poll_rx_ts);
  resp_msg_get_ts(&rx_buffer[IDX_RESP_TX_TS], &resp_tx_ts);
  int32_t rtd_init = resp_rx_ts - poll_tx_ts;
  int32_t rtd_resp = resp_tx_ts - poll_rx_ts;
  double tof = ((rtd_init - rtd_resp * (1 - clockOffsetRatio)) / 2.0) * DWT_TIME_UNITS;
  *dist = tof * SPEED_OF_LIGHT;
  return true;
}

/* 상대 앵커와 SURVEY_N번 재서 중앙값 */
static Pair measure_pair(const char *target)
{
  double d[SURVEY_N];
  int ok = 0;
  dwt_setrxaftertxdelay(POLL_TX_TO_RESP_RX_DLY_UUS);
  dwt_setrxtimeout(RESP_RX_TIMEOUT_UUS);
  for (int i = 0; i < SURVEY_N; i++)
  {
    double v;
    if (range_to(target, &v))
      d[ok++] = v;
    delay(SURVEY_GAP_MS);
  }
  dwt_setrxtimeout(0);   // 응답자로 돌아가면 타임아웃 없이 기다린다
  for (int i = 1; i < ok; i++)   // 삽입 정렬 (20개)
    for (int j = i; j > 0 && d[j] < d[j - 1]; j--)
    {
      double t = d[j]; d[j] = d[j - 1]; d[j - 1] = t;
    }
  Pair p = {target, ok ? (int32_t)lround(d[ok / 2] * 1000) : -1, (uint8_t)ok, (uint8_t)SURVEY_N};
  return p;
}

static void print_pair(const char *from, const char *to, int32_t mm, unsigned ok, unsigned tries)
{
  Serial.printf("S,%lu,%s,%s,%.3f,%u,%u\n", (unsigned long)millis(), from, to, mm < 0 ? -1.0 : mm / 1000.0, ok, tries);
}

/* 측량 결과 방송을 받으면 S 줄로 출력 */
static void print_survey_res(uint32_t frame_len)
{
  int n = rx_buffer[10];
  if (SURVEY_HDR + n * SURVEY_ENTRY + 2 > (int)frame_len)
    return;
  char from[3] = {(char)rx_buffer[IDX_SRC], (char)rx_buffer[IDX_SRC + 1], 0};
  for (int i = 0; i < n; i++)
  {
    const uint8_t *p = &rx_buffer[SURVEY_HDR + i * SURVEY_ENTRY];
    char to[3] = {(char)p[0], (char)p[1], 0};
    print_pair(from, to, (int32_t)get32(&p[2]), p[6], p[7]);
  }
}

static int anchor_index(const char *id)
{
  for (int i = 0; i < N_ANCHORS; i++)
    if (ANCHORS[i][0] == id[0] && ANCHORS[i][1] == id[1])
      return i;
  return -1;
}

/* 지휘 앵커(commander)의 지시: 나보다 목록 뒤에 있는 앵커들과 재서(지휘 앵커는 빼고) 결과를 방송 */
static void do_survey_cmd(const char *commander)
{
  Pair pairs[8];
  int n = 0;
  for (int i = anchor_index(MY_ID) + 1; i < N_ANCHORS && n < 8; i++)
    if (!(ANCHORS[i][0] == commander[0] && ANCHORS[i][1] == commander[1]))
      pairs[n++] = measure_pair(ANCHORS[i]);

  uint8_t msg[SURVEY_HDR + 8 * SURVEY_ENTRY];
  const uint8_t all[2] = {0xFF, 0xFF};
  fill_header(msg, all, FUNC_SURVEY_RES);
  msg[10] = n;
  for (int i = 0; i < n; i++)
  {
    uint8_t *p = &msg[SURVEY_HDR + i * SURVEY_ENTRY];
    p[0] = pairs[i].to[0];
    p[1] = pairs[i].to[1];
    put32(&p[2], (uint32_t)pairs[i].mm);
    p[6] = pairs[i].ok;
    p[7] = pairs[i].tries;
    print_pair(MY_ID, pairs[i].to, pairs[i].mm, pairs[i].ok, pairs[i].tries);   // 이 앵커가 컴퓨터에 꽂혀 있을 때를 위해
  }
  send_frame(msg, SURVEY_HDR + n * SURVEY_ENTRY);
}

/* 다른 앵커(from)의 측량 결과를 deadline까지 기다려 출력. 받으면 true */
static bool wait_survey_res(const char *from, uint32_t deadline)
{
  while ((int32_t)(millis() - deadline) < 0)
  {
    dwt_rxenable(DWT_START_RX_IMMEDIATE);
    uint32_t st;
    while (!((st = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG_BIT_MASK | SYS_STATUS_ALL_RX_ERR)))
    {
      if ((int32_t)(millis() - deadline) >= 0)
      {
        dwt_forcetrxoff();
        return false;
      }
    }
    if (!(st & SYS_STATUS_RXFCG_BIT_MASK))
    {
      dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_ERR);
      continue;
    }
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG_BIT_MASK);
    uint32_t len = dwt_read32bitreg(RX_FINFO_ID) & RXFLEN_MASK;
    if (len > sizeof(rx_buffer) || len <= IDX_FUNC)
      continue;
    dwt_readrxdata(rx_buffer, len, 0);
    if (rx_buffer[IDX_FUNC] == FUNC_SURVEY_RES && same_id(&rx_buffer[IDX_SRC], from))
    {
      print_survey_res(len);
      return true;
    }
    /* 기다리는 동안 온 태그 poll·방송 등은 버린다 (측량이 몇 초라 괜찮다) */
  }
  return false;
}

/* 컴퓨터의 "SURVEY" 명령 — 이 앵커가 지휘한다 */
static void run_survey()
{
  Serial.println("# survey start");
  // 1) 내가 다른 앵커 모두와 잰다
  for (int i = 0; i < N_ANCHORS; i++)
    if (anchor_index(MY_ID) != i)
    {
      Pair p = measure_pair(ANCHORS[i]);
      print_pair(MY_ID, p.to, p.mm, p.ok, p.tries);
    }
  // 2) 남은 쌍은 그 앵커에게 시킨다 (목록에서 나와 그 앵커를 뺀 뒤쪽 번호들)
  for (int i = 0; i < N_ANCHORS; i++)
  {
    if (anchor_index(MY_ID) == i)
      continue;
    bool has_work = false;
    for (int j = i + 1; j < N_ANCHORS; j++)
      if (anchor_index(MY_ID) != j)
        has_work = true;
    if (!has_work)
      continue;
    uint8_t cmd[POLL_LEN - 2];
    fill_header(cmd, (const uint8_t *)ANCHORS[i], FUNC_SURVEY_CMD);
    send_frame(cmd, sizeof(cmd));
    if (!wait_survey_res(ANCHORS[i], millis() + SURVEY_WAIT_MS))
      Serial.printf("# survey %s no answer\n", ANCHORS[i]);
  }
  Serial.println("# survey done");
}

/* 컴퓨터가 보낸 한 줄 명령 처리 */
static void handle_serial()
{
  static char line[16];
  static int n = 0;
  while (Serial.available())
  {
    char c = Serial.read();
    if (c == '\n' || c == '\r')
    {
      line[n] = 0;
      if (strcmp(line, "SURVEY") == 0)
        run_survey();
      n = 0;
    }
    else if (n < (int)sizeof(line) - 1)
      line[n++] = c;
  }
}

void setup()
{
  UART_init();

  _fastSPI = SPISettings(7000000L, MSBFIRST, SPI_MODE0);
  spiBegin(PIN_IRQ, PIN_RST);
  spiSelect(PIN_SS);
  delay(2); // DW3000 기동 대기 (INIT_RC -> IDLE_RC)

  dwt_softreset();
  delay(2);

  if (!dwt_checkidlerc())
    halt("IDLE FAILED");
  if (dwt_initialise(DWT_DW_INIT) == DWT_ERROR)
    halt("INIT FAILED");

  dwt_setleds(DWT_LEDS_ENABLE | DWT_LEDS_INIT_BLINK);

  if (dwt_configure(&config))
    halt("CONFIG FAILED");
  dwt_configuretxrf(&txconfig_options);

  dwt_setrxantennadelay(ANT_DLY);
  dwt_settxantennadelay(ANT_DLY);
  dwt_setlnapamode(DWT_LNA_ENABLE | DWT_PA_ENABLE);

  uint8_t mac[6];
  esp_read_mac(mac, ESP_MAC_WIFI_STA);
  Serial.printf("# uwb_anchor id=%s mac=%02x:%02x:%02x:%02x:%02x:%02x ant_dly=%d dev_id=%08lx\n",
                MY_ID, mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], ANT_DLY, (unsigned long)dwt_readdevid());
}

void loop()
{
  uint32_t status_reg;

  /* 즉시 수신 시작. 타임아웃 없이 기다리되, 그 사이 컴퓨터 명령이 오면 수신을 멈추고 처리한다 */
  dwt_rxenable(DWT_START_RX_IMMEDIATE);
  while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG_BIT_MASK | SYS_STATUS_ALL_RX_ERR)))
  {
    if (Serial.available())
    {
      dwt_forcetrxoff();
      handle_serial();
      return;
    }
  };

  if (status_reg & SYS_STATUS_RXFCG_BIT_MASK)
  {
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG_BIT_MASK);
    uint32_t frame_len = dwt_read32bitreg(RX_FINFO_ID) & RXFLEN_MASK;
    if (frame_len <= sizeof(rx_buffer) && frame_len > IDX_FUNC)
    {
      dwt_readrxdata(rx_buffer, frame_len, 0);
      uint8_t func = rx_buffer[IDX_FUNC];

      /* 내 번호로 온 poll만 답한다. 다른 앵커 앞으로 온 poll, 다른 앵커의 응답 등은 무시 */
      if (func == FUNC_POLL && frame_len == POLL_LEN && is_me(&rx_buffer[IDX_DST]))
        respond_to_poll();
      /* 태그의 바퀴별 결과 방송 — 모든 앵커가 받아 출력한다 (컴퓨터에 꽂힌 앵커의 출력만 쓰인다) */
      else if (func == FUNC_REPORT && frame_len >= REPORT_HDR + 2)
        print_tag_report(frame_len);
      /* 측량 지시 (나에게) */
      else if (func == FUNC_SURVEY_CMD && is_me(&rx_buffer[IDX_DST]))
      {
        char commander[3] = {(char)rx_buffer[IDX_SRC], (char)rx_buffer[IDX_SRC + 1], 0};
        do_survey_cmd(commander);
      }
      /* 다른 앵커의 측량 결과 — 컴퓨터에 꽂힌 앵커라면 화면에 보이도록 출력 */
      else if (func == FUNC_SURVEY_RES)
        print_survey_res(frame_len);
    }
  }
  else
  {
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_ERR);
  }

  report();
}
