/*
 * uwb_anchor — SS-TWR 응답자(앵커). 태그(uwb_tag)의 poll을 받으면 타임스탬프를 담아 응답한다.
 * 거리 계산은 태그 쪽에서 한다. 앵커는 전원만 있으면 되고 PC 연결은 필요 없다.
 *
 * 출처: Makerfabs-ESP32-UWB-DW3000 example/range/range_tx (Qorvo ex_06b 기반)
 * 바꾼 것:
 *   - 부팅 시 칩 ID·안테나 지연을 '#' 주석 줄로 출력 (보드 식별용)
 *   - 1초마다 응답 횟수를 출력 → 태그와 통신이 되고 있는지 앵커 쪽에서도 보임
 *
 * 출력 (115200bps):
 *   # ...                     부팅 정보
 *   # replies=<n> per 1s      지난 1초 동안 응답한 횟수 (트래픽이 있을 때만)
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
// Ranging — 태그와 반드시 같은 값이어야 하는 것들
// ======================================================
#define TX_ANT_DLY 16385   // 안테나 지연 (보정 전 기본값). 보정 후 보드별로 바꾼다
#define RX_ANT_DLY 16385
#define ALL_MSG_COMMON_LEN 10
#define ALL_MSG_SN_IDX 2
#define RESP_MSG_POLL_RX_TS_IDX 10
#define RESP_MSG_RESP_TX_TS_IDX 14
#define RESP_MSG_TS_LEN 4
#define POLL_RX_TO_RESP_TX_DLY_UUS 900   // poll 수신 → 응답 송신까지 (태그의 대기시간과 짝)

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

static uint8_t rx_poll_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'W', 'A', 'V', 'E', 0xE0, 0, 0};
static uint8_t tx_resp_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'V', 'E', 'W', 'A', 0xE1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static uint8_t frame_seq_nb = 0;
static uint8_t rx_buffer[20];
static uint32_t status_reg = 0;
static uint64_t poll_rx_ts;
static uint64_t resp_tx_ts;

static uint32_t replies = 0;
static uint32_t last_report_ms = 0;

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

  dwt_setlnapamode(DWT_LNA_ENABLE | DWT_PA_ENABLE);

  Serial.printf("# uwb_anchor dev_id=%08lx tx_ant_dly=%d rx_ant_dly=%d\n",
                (unsigned long)dwt_readdevid(), TX_ANT_DLY, RX_ANT_DLY);
}

void loop()
{
  /* 즉시 수신 시작. 타임아웃 없이 poll이 올 때까지 기다린다 */
  dwt_rxenable(DWT_START_RX_IMMEDIATE);

  while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG_BIT_MASK | SYS_STATUS_ALL_RX_ERR)))
  {
  };

  if (status_reg & SYS_STATUS_RXFCG_BIT_MASK)
  {
    uint32_t frame_len;

    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG_BIT_MASK);

    frame_len = dwt_read32bitreg(RX_FINFO_ID) & RXFLEN_MASK;
    if (frame_len <= sizeof(rx_buffer))
    {
      dwt_readrxdata(rx_buffer, frame_len, 0);

      /* 시퀀스 번호는 비교에서 빼고, 태그가 보낸 poll인지만 확인 */
      rx_buffer[ALL_MSG_SN_IDX] = 0;
      if (memcmp(rx_buffer, rx_poll_msg, ALL_MSG_COMMON_LEN) == 0)
      {
        uint32_t resp_tx_time;
        int ret;

        poll_rx_ts = get_rx_timestamp_u64();

        /* 응답은 poll 수신 시각 + 고정 지연에 예약 송신. 그래야 송신 시각을 미리 알고 메시지에 담을 수 있다 */
        resp_tx_time = (poll_rx_ts + (POLL_RX_TO_RESP_TX_DLY_UUS * UUS_TO_DWT_TIME)) >> 8;
        dwt_setdelayedtrxtime(resp_tx_time);

        /* 실제 송신 타임스탬프 = 예약 시각 + 안테나 지연 */
        resp_tx_ts = (((uint64_t)(resp_tx_time & 0xFFFFFFFEUL)) << 8) + TX_ANT_DLY;

        resp_msg_set_ts(&tx_resp_msg[RESP_MSG_POLL_RX_TS_IDX], poll_rx_ts);
        resp_msg_set_ts(&tx_resp_msg[RESP_MSG_RESP_TX_TS_IDX], resp_tx_ts);

        tx_resp_msg[ALL_MSG_SN_IDX] = frame_seq_nb;
        dwt_writetxdata(sizeof(tx_resp_msg), tx_resp_msg, 0);
        dwt_writetxfctrl(sizeof(tx_resp_msg), 0, 1);
        ret = dwt_starttx(DWT_START_TX_DELAYED);

        /* 예약 시각을 이미 지나쳤으면 실패 — 이번 교환은 버리고 다음 poll을 기다린다 */
        if (ret == DWT_SUCCESS)
        {
          while (!(dwt_read32bitreg(SYS_STATUS_ID) & SYS_STATUS_TXFRS_BIT_MASK))
          {
          };
          dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS_BIT_MASK);
          frame_seq_nb++;
          replies++;
        }
      }
    }
  }
  else
  {
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_ERR);
  }

  uint32_t now = millis();
  if (now - last_report_ms >= 1000)
  {
    if (replies > 0)
      Serial.printf("# replies=%lu per 1s\n", (unsigned long)replies);
    replies = 0;
    last_report_ms = now;
  }
}
