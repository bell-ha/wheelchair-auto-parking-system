/*
 * uwb_tag — 태그 (휠체어에 다는 쪽). 앵커들에게 차례로 poll을 보내고, 응답에 담긴 타임스탬프로 거리를 계산한다.
 * PC에 USB로 연결해 tools/live_view.py로 보고 기록한다.
 *
 * 업로드 전에 아래 "보드마다 바꾸는 값" 두 줄만 확인한다.
 *
 * 출처: Makerfabs-ESP32-UWB-DW3000 example/range/range_rx (Qorvo ex_06a 기반)
 * 바꾼 것:
 *   - 번호(MY_ID): 앵커 A1, A2에게 번호를 붙여 차례로 묻는다 → 거리 2개가 섞이지 않는다
 *   - 태그 2개가 같은 순간에 말하지 않게 주기를 매번 조금씩 흔든다
 *   - 계속 답이 없는 앵커(꺼졌거나 멂)는 1초에 한 번만 다시 묻는다
 *   - "DIST: x.xx m" 대신 한 줄 CSV로 출력 → live_view.py가 그대로 저장
 *   - 수신 진단값(C, F1~F3, N, D)을 원시값으로 출력. dBm 환산과 가림 판정은 PC에서 한다
 *   - 실패(timeout·rx_err·bad_frame)도 한 줄씩 남긴다 → 가려졌을 때 성공률을 잴 수 있다
 *
 * 무선 메시지 — uwb_anchor.ino 머리말 참고 (주소 2바이트 = 번호 글자 2개)
 *
 * 출력 (115200bps):
 *   # uwb_tag id=T1 mac=.. ant_dly=.. period_ms=..             부팅 정보
 *   R,ms,seq,tag,anchor,dist_m,clk_ppm,C,F1,F2,F3,N,D          측정 성공
 *   E,ms,seq,tag,anchor,reason                                 측정 실패 (timeout | rx_err | bad_frame)
 *
 * 거리 공식 (SS-TWR, 시계 차이 보정) — docs/UWB_설명.txt 3절
 *   tof = (Rtd_init - Rtd_resp * (1 - clk_offset)) / 2
 *   Rtd_init = 태그가 잰 왕복시간, Rtd_resp = 앵커가 잰 응답 대기시간
 */
#include "dw3000.h"
#include "SPI.h"
#include <esp_system.h>

extern SPISettings _fastSPI;

// ======================================================
// 보드마다 바꾸는 값 — 업로드 전에 확인
// ======================================================
#define MY_ID "T1"        // 첫 번째 태그 "T1", 두 번째 태그 "T2". 두 보드에 같은 번호를 넣으면 안 된다
#define ANT_DLY 16385     // 이 보드의 안테나 지연. 보정 전 기본값 16385 (보정값은 docs/UWB_설명.txt 2절 보드 목록)

// 차례로 물어볼 앵커 번호 (앵커 코드의 MY_ID와 같아야 한다)
static const char *ANCHORS[] = {"A1", "A2"};

// ======================================================
// Pin map — 보드 내부 배선 (ESP32 <-> DW3000). 바꾸지 말 것
// ======================================================
#define PIN_RST 27
#define PIN_IRQ 34
#define PIN_SS 4

// ======================================================
// Ranging — 앵커 코드와 반드시 같은 값이어야 하는 것들
// ======================================================
#define CYCLE_MS 100                     // 앵커 전부를 한 바퀴 묻는 주기 (10 Hz)
#define JITTER_MS 15                     // 태그 2개가 같은 순간에 말하지 않게 주기를 매번 ±15 ms 흔든다
#define ABSENT_AFTER 10                  // 연속으로 이만큼 실패한 앵커는
#define ABSENT_RETRY_MS 1000             //   이 간격으로만 다시 묻는다
#define POLL_TX_TO_RESP_RX_DLY_UUS 600   // poll 보내고 이만큼 쉬었다가 수신기를 켠다 (앵커의 900과 짝)
#define RESP_RX_TIMEOUT_UUS 400          // 그 뒤 이 시간 안에 응답이 없으면 timeout

#define FUNC_POLL 0xE0
#define FUNC_RESP 0xE1
#define IDX_SEQ 2
#define IDX_DST 5          // 받는 쪽 번호 2바이트
#define IDX_SRC 7          // 보내는 쪽 번호 2바이트
#define IDX_FUNC 9
#define IDX_POLL_RX_TS 10
#define IDX_RESP_TX_TS 14
#define POLL_LEN 12        // CRC 포함
#define RESP_LEN 20

// DGC_DBG (0x03:0x60) [30:28] = DGC_DECISION. 수신 이득 단계로, dBm 환산에 6 dB/단계로 더해진다.
// 이 라이브러리 regs.h에는 정의가 없어 DW3000 User Manual의 주소를 직접 쓴다 (아직 실측 검증 안 함)
#define DGC_DBG_ID 0x30060

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
static uint8_t rx_buffer[RESP_LEN];
static uint32_t seq = 0;                       // CSV용 일련번호 (frame_seq_nb는 256에서 돌아간다)
static uint32_t fail_streak[N_ANCHORS] = {0};
static uint32_t next_try_ms[N_ANCHORS] = {0};
static dwt_rxdiag_t diag;

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

/* 앵커 하나에 한 번 묻고 결과를 한 줄 출력. 성공하면 true */
static bool range_once(const char *anchor)
{
  uint8_t tx_poll[POLL_LEN - 2] = {0x41, 0x88, 0, 0xCA, 0xDE};   // CRC 2바이트는 칩이 붙인다
  tx_poll[IDX_SEQ] = frame_seq_nb;
  tx_poll[IDX_DST] = anchor[0];
  tx_poll[IDX_DST + 1] = anchor[1];
  tx_poll[IDX_SRC] = MY_ID[0];
  tx_poll[IDX_SRC + 1] = MY_ID[1];
  tx_poll[IDX_FUNC] = FUNC_POLL;

  dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS_BIT_MASK);
  dwt_writetxdata(sizeof(tx_poll), tx_poll, 0);
  dwt_writetxfctrl(sizeof(tx_poll) + 2, 0, 1);

  /* 송신 후 setrxaftertxdelay만큼 지나면 수신기가 자동으로 켜진다 */
  dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);

  uint32_t status_reg;
  while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG_BIT_MASK | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR)))
  {
  };

  frame_seq_nb++;
  seq++;

  if (!(status_reg & SYS_STATUS_RXFCG_BIT_MASK))
  {
    const char *reason = (status_reg & SYS_STATUS_ALL_RX_TO) ? "timeout" : "rx_err";
    Serial.printf("E,%lu,%lu,%s,%s,%s\n", (unsigned long)millis(), (unsigned long)seq, MY_ID, anchor, reason);
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
    return false;
  }

  dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG_BIT_MASK);
  uint32_t frame_len = dwt_read32bitreg(RX_FINFO_ID) & RXFLEN_MASK;
  bool ok = false;
  if (frame_len == RESP_LEN)
  {
    dwt_readrxdata(rx_buffer, frame_len, 0);
    /* 내가 물어본 앵커가 나에게 보낸 응답인지 확인 (다른 태그 앞으로 간 응답이 섞여 들어올 수 있다) */
    ok = rx_buffer[IDX_FUNC] == FUNC_RESP && same_id(&rx_buffer[IDX_DST], MY_ID) && same_id(&rx_buffer[IDX_SRC], anchor);
  }
  if (!ok)
  {
    Serial.printf("E,%lu,%lu,%s,%s,bad_frame\n", (unsigned long)millis(), (unsigned long)seq, MY_ID, anchor);
    return false;
  }

  uint32_t poll_tx_ts = dwt_readtxtimestamplo32();
  uint32_t resp_rx_ts = dwt_readrxtimestamplo32();
  float clockOffsetRatio = ((float)dwt_readclockoffset()) / (uint32_t)(1 << 26);
  uint32_t poll_rx_ts, resp_tx_ts;
  resp_msg_get_ts(&rx_buffer[IDX_POLL_RX_TS], &poll_rx_ts);
  resp_msg_get_ts(&rx_buffer[IDX_RESP_TX_TS], &resp_tx_ts);

  int32_t rtd_init = resp_rx_ts - poll_tx_ts;
  int32_t rtd_resp = resp_tx_ts - poll_rx_ts;
  double tof = ((rtd_init - rtd_resp * (1 - clockOffsetRatio)) / 2.0) * DWT_TIME_UNITS;
  double distance = tof * SPEED_OF_LIGHT;

  /* 같은 수신 프레임의 진단값. 다음 송수신 전에 읽어야 한다 */
  dwt_readdiagnostics(&diag);
  uint8_t dgc = (dwt_read32bitreg(DGC_DBG_ID) >> 28) & 0x7;

  Serial.printf("R,%lu,%lu,%s,%s,%.3f,%.2f,%lu,%lu,%lu,%lu,%u,%u\n",
                (unsigned long)millis(), (unsigned long)seq, MY_ID, anchor, distance, clockOffsetRatio * 1e6,
                (unsigned long)diag.ipatovPower, (unsigned long)diag.ipatovF1,
                (unsigned long)diag.ipatovF2, (unsigned long)diag.ipatovF3,
                (unsigned)diag.ipatovAccumCount, (unsigned)dgc);
  return true;
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

  /* 응답 대기 시간과 타임아웃은 매번 같으므로 한 번만 설정 */
  dwt_setrxaftertxdelay(POLL_TX_TO_RESP_RX_DLY_UUS);
  dwt_setrxtimeout(RESP_RX_TIMEOUT_UUS);

  dwt_setlnapamode(DWT_LNA_ENABLE | DWT_PA_ENABLE);

  /* 진단 레지스터 전체 기록. 꺼져 있으면 C·F1~F3가 채워지지 않는다 */
  dwt_configciadiag(DW_CIA_DIAG_LOG_ALL);

  randomSeed(esp_random());

  uint8_t mac[6];
  esp_read_mac(mac, ESP_MAC_WIFI_STA);
  Serial.printf("# uwb_tag id=%s mac=%02x:%02x:%02x:%02x:%02x:%02x ant_dly=%d period_ms=%d dev_id=%08lx\n",
                MY_ID, mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], ANT_DLY, CYCLE_MS, (unsigned long)dwt_readdevid());
  Serial.println("# R,ms,seq,tag,anchor,dist_m,clk_ppm,C,F1,F2,F3,N,D");
  Serial.println("# E,ms,seq,tag,anchor,reason");
}

void loop()
{
  uint32_t start = millis();
  for (int i = 0; i < N_ANCHORS; i++)
  {
    /* 계속 실패하는 앵커(꺼졌거나 너무 멂)는 1초에 한 번만 다시 묻는다 */
    if (fail_streak[i] >= ABSENT_AFTER && (int32_t)(millis() - next_try_ms[i]) < 0)
      continue;
    if (range_once(ANCHORS[i]))
      fail_streak[i] = 0;
    else if (++fail_streak[i] >= ABSENT_AFTER)
      next_try_ms[i] = millis() + ABSENT_RETRY_MS;
  }
  int32_t wait = CYCLE_MS - (int32_t)(millis() - start) + (int32_t)random(-JITTER_MS, JITTER_MS + 1);
  if (wait > 0)
    delay(wait);
}
