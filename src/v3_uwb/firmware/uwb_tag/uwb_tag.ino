/*
 * uwb_tag — SS-TWR 개시자(태그). 앵커(uwb_anchor)에 poll을 보내고, 응답에 담긴 타임스탬프로 거리를 계산한다.
 * PC에 USB로 연결해 tools/log_range.py로 기록한다.
 *
 * 출처: Makerfabs-ESP32-UWB-DW3000 example/range/range_rx (Qorvo ex_06a 기반)
 * 바꾼 것:
 *   - "DIST: x.xx m" 대신 한 줄 CSV로 출력 → log_range.py가 그대로 저장
 *   - 수신 진단값(C, F1~F3, N, D)을 원시값으로 출력. dBm 환산과 NLOS 판정은 PC에서 한다
 *     (공식이 틀렸으면 재업로드 없이 tools/만 고치면 된다)
 *   - 실패(타임아웃·수신오류·엉뚱한 프레임)도 한 줄씩 남김 → 차체에 가렸을 때 성공률을 잴 수 있다
 *   - 측정 주기 1 Hz → 10 Hz
 *
 * 출력 (115200bps):
 *   # ...                                     부팅 정보
 *   R,ms,seq,dist_m,clk_ppm,C,F1,F2,F3,N,D    측정 성공
 *   E,ms,seq,reason                           측정 실패 (timeout | rx_err | bad_frame)
 *
 * 거리 공식 (SS-TWR, 클럭 오프셋 보정):
 *   tof = (Rtd_init - Rtd_resp * (1 - clk_offset)) / 2
 *   Rtd_init = 태그가 잰 왕복시간, Rtd_resp = 앵커가 잰 응답 대기시간
 *   clk_offset는 수신 신호의 반송파로 추정한 앵커 대비 태그 시계 오차
 */
#include "dw3000.h"
#include "SPI.h"

extern SPISettings _fastSPI;

// ======================================================
// Pin map — 보드 내부 배선 (ESP32 <-> DW3000). 바꾸지 말 것
// ======================================================
#define PIN_RST 27
#define PIN_IRQ 34
#define PIN_SS 4

// ======================================================
// Ranging — 앵커와 반드시 같은 값이어야 하는 것들
// ======================================================
#define RNG_DELAY_MS 100   // 측정 주기 (10 Hz)
#define TX_ANT_DLY 16385   // 안테나 지연 (보정 전 기본값). 보정 후 보드별로 바꾼다
#define RX_ANT_DLY 16385
#define ALL_MSG_COMMON_LEN 10
#define ALL_MSG_SN_IDX 2
#define RESP_MSG_POLL_RX_TS_IDX 10
#define RESP_MSG_RESP_TX_TS_IDX 14
#define RESP_MSG_TS_LEN 4
#define POLL_TX_TO_RESP_RX_DLY_UUS 600   // poll 송신 후 이만큼 쉬었다가 수신기를 켠다
#define RESP_RX_TIMEOUT_UUS 400          // 그 뒤 이 시간 안에 응답이 없으면 timeout

// DGC_DBG (0x03:0x60) [30:28] = DGC_DECISION. 수신 이득 단계로, dBm 환산에 6 dB/단계로 더해진다.
// 이 라이브러리 regs.h에는 정의가 없어 DW3000 User Manual의 주소를 직접 쓴다.
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

static uint8_t tx_poll_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'W', 'A', 'V', 'E', 0xE0, 0, 0};
static uint8_t rx_resp_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'V', 'E', 'W', 'A', 0xE1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static uint8_t frame_seq_nb = 0;
static uint8_t rx_buffer[20];
static uint32_t status_reg = 0;
static uint32_t seq = 0;   // CSV용 일련번호 (frame_seq_nb는 256에서 돌아간다)
static dwt_rxdiag_t diag;

extern dwt_txconfig_t txconfig_options;

static void halt(const char *msg)
{
  Serial.printf("# FATAL %s\n", msg);
  while (1)
    ;
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

  dwt_setrxantennadelay(RX_ANT_DLY);
  dwt_settxantennadelay(TX_ANT_DLY);

  /* 응답 대기 시간과 타임아웃은 매번 같으므로 한 번만 설정 */
  dwt_setrxaftertxdelay(POLL_TX_TO_RESP_RX_DLY_UUS);
  dwt_setrxtimeout(RESP_RX_TIMEOUT_UUS);

  dwt_setlnapamode(DWT_LNA_ENABLE | DWT_PA_ENABLE);

  /* 진단 레지스터 전체 기록. 꺼져 있으면 C·F1~F3가 채워지지 않는다 */
  dwt_configciadiag(DW_CIA_DIAG_LOG_ALL);

  Serial.printf("# uwb_tag dev_id=%08lx tx_ant_dly=%d rx_ant_dly=%d period_ms=%d\n",
                (unsigned long)dwt_readdevid(), TX_ANT_DLY, RX_ANT_DLY, RNG_DELAY_MS);
  Serial.println("# R,ms,seq,dist_m,clk_ppm,C,F1,F2,F3,N,D");
  Serial.println("# E,ms,seq,reason");
}

void loop()
{
  tx_poll_msg[ALL_MSG_SN_IDX] = frame_seq_nb;
  dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS_BIT_MASK);
  dwt_writetxdata(sizeof(tx_poll_msg), tx_poll_msg, 0);
  dwt_writetxfctrl(sizeof(tx_poll_msg), 0, 1);

  /* 송신 후 setrxaftertxdelay만큼 지나면 수신기가 자동으로 켜진다 */
  dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);

  while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG_BIT_MASK | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR)))
  {
  };

  frame_seq_nb++;
  seq++;

  if (status_reg & SYS_STATUS_RXFCG_BIT_MASK)
  {
    uint32_t frame_len;

    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG_BIT_MASK);

    frame_len = dwt_read32bitreg(RX_FINFO_ID) & RXFLEN_MASK;
    bool ok = false;
    if (frame_len <= sizeof(rx_buffer))
    {
      dwt_readrxdata(rx_buffer, frame_len, 0);

      /* 시퀀스 번호는 비교에서 빼고, 앵커의 응답인지만 확인 */
      rx_buffer[ALL_MSG_SN_IDX] = 0;
      if (memcmp(rx_buffer, rx_resp_msg, ALL_MSG_COMMON_LEN) == 0)
      {
        uint32_t poll_tx_ts, resp_rx_ts, poll_rx_ts, resp_tx_ts;
        int32_t rtd_init, rtd_resp;
        float clockOffsetRatio;

        poll_tx_ts = dwt_readtxtimestamplo32();
        resp_rx_ts = dwt_readrxtimestamplo32();

        clockOffsetRatio = ((float)dwt_readclockoffset()) / (uint32_t)(1 << 26);

        resp_msg_get_ts(&rx_buffer[RESP_MSG_POLL_RX_TS_IDX], &poll_rx_ts);
        resp_msg_get_ts(&rx_buffer[RESP_MSG_RESP_TX_TS_IDX], &resp_tx_ts);

        rtd_init = resp_rx_ts - poll_tx_ts;
        rtd_resp = resp_tx_ts - poll_rx_ts;

        double tof = ((rtd_init - rtd_resp * (1 - clockOffsetRatio)) / 2.0) * DWT_TIME_UNITS;
        double distance = tof * SPEED_OF_LIGHT;

        /* 같은 수신 프레임의 진단값. 다음 송수신 전에 읽어야 한다 */
        dwt_readdiagnostics(&diag);
        uint8_t dgc = (dwt_read32bitreg(DGC_DBG_ID) >> 28) & 0x7;

        Serial.printf("R,%lu,%lu,%.3f,%.2f,%lu,%lu,%lu,%lu,%u,%u\n",
                      (unsigned long)millis(), (unsigned long)seq, distance, clockOffsetRatio * 1e6,
                      (unsigned long)diag.ipatovPower, (unsigned long)diag.ipatovF1,
                      (unsigned long)diag.ipatovF2, (unsigned long)diag.ipatovF3,
                      (unsigned)diag.ipatovAccumCount, (unsigned)dgc);
        ok = true;
      }
    }
    if (!ok)
      Serial.printf("E,%lu,%lu,bad_frame\n", (unsigned long)millis(), (unsigned long)seq);
  }
  else
  {
    const char *reason = (status_reg & SYS_STATUS_ALL_RX_TO) ? "timeout" : "rx_err";
    Serial.printf("E,%lu,%lu,%s\n", (unsigned long)millis(), (unsigned long)seq, reason);
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
  }

  Sleep(RNG_DELAY_MS);
}
