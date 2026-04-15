import smbus2
import time

class MAX30102:
    # Register map
    REG_INT_STATUS_1  = 0x00
    REG_FIFO_WR_PTR   = 0x04
    REG_OVF_COUNTER   = 0x05
    REG_FIFO_RD_PTR   = 0x06
    REG_FIFO_DATA     = 0x07
    REG_FIFO_CONFIG   = 0x08
    REG_MODE_CONFIG   = 0x09
    REG_SPO2_CONFIG   = 0x0A
    REG_LED1_PA       = 0x0C   # RED pulse amplitude
    REG_LED2_PA       = 0x0D   # IR  pulse amplitude

    def __init__(self, address=0x57):
        self.bus = smbus2.SMBus(1)
        self.address = address

        # Software Reset (bit 6 of REG_MODE_CONFIG)
        self.bus.write_byte_data(self.address, self.REG_MODE_CONFIG, 0x40)
        time.sleep(1)

        # Clear FIFO pointers
        self.bus.write_byte_data(self.address, self.REG_FIFO_WR_PTR,  0x00)
        self.bus.write_byte_data(self.address, self.REG_OVF_COUNTER,   0x00)
        self.bus.write_byte_data(self.address, self.REG_FIFO_RD_PTR,  0x00)

        # FIFO config: SMP_AVE=1 (no averaging), FIFO_ROLLOVER_EN=1, FIFO_A_FULL=15
        self.bus.write_byte_data(self.address, self.REG_FIFO_CONFIG, 0x1F)

        # Mode: SpO2 (RED + IR)
        self.bus.write_byte_data(self.address, self.REG_MODE_CONFIG, 0x03)

        # SpO2 config: ADC range=4096nA, SR=100sps, LED pw=411us (18-bit)
        self.bus.write_byte_data(self.address, self.REG_SPO2_CONFIG, 0x27)

        # LED pulse amplitude (~7.2mA each)
        self.bus.write_byte_data(self.address, self.REG_LED1_PA, 0x24)  # RED
        self.bus.write_byte_data(self.address, self.REG_LED2_PA, 0x24)  # IR

    def _num_samples_available(self):
        """Return how many unread samples are in the FIFO."""
        wr = self.bus.read_byte_data(self.address, self.REG_FIFO_WR_PTR) & 0x1F
        rd = self.bus.read_byte_data(self.address, self.REG_FIFO_RD_PTR) & 0x1F
        return (wr - rd) & 0x1F

    def read_fifo(self):
        """
        Read the latest RED and IR sample.
        Returns (red, ir) as 18-bit integers.
        Returns (0, 0) if no new data is available.
        """
        n = self._num_samples_available()
        if n == 0:
            return 0, 0

        # Drain all but the most recent sample to avoid falling behind
        # (each sample = 6 bytes: 3 RED + 3 IR)
        for _ in range(n - 1):
            self.bus.read_i2c_block_data(self.address, self.REG_FIFO_DATA, 6)

        data = self.bus.read_i2c_block_data(self.address, self.REG_FIFO_DATA, 6)
        red = (data[0] << 16 | data[1] << 8 | data[2]) & 0x03FFFF
        ir  = (data[3] << 16 | data[4] << 8 | data[5]) & 0x03FFFF

        return red, ir
