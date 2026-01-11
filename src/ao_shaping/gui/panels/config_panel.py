    def load_config_from_data(self, config_data: dict):
        """从数据加载配置"""
        try:
            # 更新控件值
            if 'N' in config_data:
                self.controls['N'].setValue(config_data['N'])
            if 'L' in config_data:
                self.controls['L'].setValue(config_data['L'])
            if 'wavelength' in config_data:
                self.controls['wavelength'].setValue(config_data['wavelength'])

            if 'Cn2' in config_data:
                self.controls['Cn2'].setValue(config_data['Cn2'])
            if 'L0' in config_data:
                self.controls['L0'].setValue(config_data['L0'])
            if 'l0' in config_data:
                self.controls['l0'].setValue(config_data['l0'])

            if 'dm_actuators' in config_data:
                self.controls['dm_actuators'].setValue(config_data['dm_actuators'])
            if 'dm_stroke' in config_data:
                self.controls['dm_stroke'].setValue(config_data['dm_stroke'])
            if 'dm_infill' in config_data:
                self.controls['dm_infill'].setChecked(config_data['dm_infill'])

            if 'subapertures' in config_data:
                self.controls['subapertures'].setValue(config_data['subapertures'])
            if 'pixel_scale' in config_data:
                self.controls['pixel_scale'].setValue(config_data['pixel_scale'])

            if 'propagation_distance' in config_data:
                self.controls['propagation_distance'].setValue(config_data['propagation_distance'])

        except Exception as e:
            print(f"从数据加载配置时出错: {e}")