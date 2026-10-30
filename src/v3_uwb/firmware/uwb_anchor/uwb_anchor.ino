/*
 * uwb_anchor — 앵커 (차에 다는 쪽). 자기 번호로 온 poll에만 답한다 (SS-TWR 응답자).
 * 거리 계산은 태그가 한다. 앵커는 전원만 있으면 되고 PC 연결은 필요 없다.
 *
 * 업로드 전에 아래 "보드마다 바꾸는 값" 두 줄만 확인한다.
 *
 * 출처: Makerfabs-ESP32-UWB-DW3000 example/range/range_tx (Qorvo ex_06b 기반)
 * 바꾼 것:
 *   - 번호(MY_ID): 자기 번호로 온 poll에만 답하고, 물어본 태그에게 답한다 → 앵커 여러 개를 같이 켤 수 있다
 *   - 부팅 시 번호·MAC·안테나 지연을 '#' 줄로 출력 (보드 확인용)
 *   - 1초마다 태그별 응답 횟수 출력
 *
 * 무선 메시지 — IEEE 802.15.4 데이터 프레임 모양. 주소 2바이트에 번호 글자 2개("A1", "T2")를 넣는다
 *   poll      41 88 seq CA DE | 받는쪽 2 | 보내는쪽 2 | E0                                   (+CRC 2 = 12바이트)
 *   response  41 88 seq CA DE | 받는쪽 2 | 보내는쪽 2 | E1 | poll 받은 시각 4 | 응답 보낸 시각 4 (+CRC 2 = 20바이트)
 *
 * 출력 (115200bps):
 *   # uwb_anchor id=A1 mac=.. ant_dly=.. dev_id=..    부팅 정보
 *   # replies=<합계> per 1s (T1=.. T2=..)             지난 1초 동안 응답한 횟수 (요청이 있을 때만)
 */
#include "dw3000.h"
#include "SPI.h"
#include <esp_system.h>

extern SPISettings _fastSPI;

// ======================================================
// 보드마다 바꾸는 값 — 업로드 전에 확인
// ======================================================
#define MY_ID "A1"        // 첫 번째 앵커 "A1", 두 번째 앵커 "A2". 두 보드에 같은 번호를 넣으면 안 된다
#define ANT_DLY 16385     // 이 보드의 안테나 지연. 보정 전 기본값 16385 (보정값은 docs/UWB_설명.txt 2절 보드 목록)

// ======================================================
// Pin map — 보드 내부 배선 (ESP32 <-> DW3000). 바꾸지 말 것
// ======================================================
#define PIN_RST 27
#define PIN_IRQ 34
#define PIN_SS 4

// ======================================================
// Ranging — 태그 코드와 반드시 같은 값이어야 하는 것들
// ======================================================
#define POLL_RX_TO_RESP_TX_DLY_UUS 900   // poll 받고 응답 보내기까지 (태그의 대기시간과 짝)

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

static uint8_t frame_seq_nb = 0;
static uint8_t rx_buffer[POLL_LEN];

// 태그별 응답 횟수 (1초마다 출력하고 비운다). 처음 보는 태그 번호를 차례로 칸에 넣는다
#define MAX_TAGS 4
static char tag_ids[MAX_TAGS][3];
static uint32_t tag_replies[MAX_TAGS];
static int n_tags = 0;
static uint32_t replies_total = 0;
static uint32_t last_report_ms = 0;

static void halt(const char *msg)
{
  Serial.printf("# FATAL %s\n", msg);
  while (1)
    ;
}

static bool is_me(const uint8_t *field)
{
  return field[0] == (uint8_t)MY_ID[0] && field[1] == (uint8_t)MY_ID[1];
}

static void count_reply(const uint8_t *src)
{
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

  /* 즉시 수신 시작. 타임아웃 없이 poll이 올 때까지 기다린다 */
  dwt_rxenable(DWT_START_RX_IMMEDIATE);
  while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG_BIT_MASK | SYS_STATUS_ALL_RX_ERR)))
  {
  };

  if (status_reg & SYS_STATUS_RXFCG_BIT_MASK)
  {
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG_BIT_MASK);
    uint32_t frame_len = dwt_read32bitreg(RX_FINFO_ID) & RXFLEN_MASK;

    /* 내 번호로 온 poll만 답한다. 다른 앵커 앞으로 온 poll, 다른 앵커의 응답 등은 무시 */
    if (frame_len == POLL_LEN)
    {
      dwt_readrxdata(rx_buffer, frame_len, 0);
      if (rx_buffer[IDX_FUNC] == FUNC_POLL && is_me(&rx_buffer[IDX_DST]))
      {
        uint8_t tx_resp[RESP_LEN - 2] = {0x41, 0x88, 0, 0xCA, 0xDE};   // CRC 2바이트는 칩이 붙인다
        uint64_t poll_rx_ts = get_rx_timestamp_u64();

        /* 응답은 poll 수신 시각 + 고정 지연에 예약 송신. 그래야 송신 시각을 미리 알고 메시지에 담을 수 있다 */
        uint32_t resp_tx_time = (poll_rx_ts + (POLL_RX_TO_RESP_TX_DLY_UUS * UUS_TO_DWT_TIME)) >> 8;
        dwt_setdelayedtrxtime(resp_tx_time);
        /* 실제 송신 타임스탬프 = 예약 시각 + 안테나 지연 */
        uint64_t resp_tx_ts = (((uint64_t)(resp_tx_time & 0xFFFFFFFEUL)) << 8) + ANT_DLY;

        tx_resp[IDX_SEQ] = frame_seq_nb;
        tx_resp[IDX_DST] = rx_buffer[IDX_SRC];         // 물어본 태그에게
        tx_resp[IDX_DST + 1] = rx_buffer[IDX_SRC + 1];
        tx_resp[IDX_SRC] = MY_ID[0];
        tx_resp[IDX_SRC + 1] = MY_ID[1];
        tx_resp[IDX_FUNC] = FUNC_RESP;
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
          frame_seq_nb++;
          count_reply(&rx_buffer[IDX_SRC]);
        }
      }
    }
  }
  else
  {
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_ERR);
  }

  report();
}
