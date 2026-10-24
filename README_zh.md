# 面向说话人安全的语音命令流水线

这是一个与具体模型实现解耦的参考工程，说明如何把多个开源语音模型组合成“重叠感知、说话人门控、低分保守救援”的完整命令识别流水线。

工程重点是模型之间的组合方式：统一的模型适配器接口、候选音频路由、保守救援判定、自动测试和可直接运行的模拟示例。实际模型通过小型适配器接入，因此可以替换具体实现，而不用改写整套决策逻辑。

## 核心思路

```text
唤醒音
  -> 重叠检测
  -> 永远保留原音候选
  -> 若重叠：双路分离 -> 增强
  -> 轻量 ASR + 唤醒文本软匹配
  -> 选择注册音

命令音
  -> 重叠检测
  -> 若重叠：
       原音 +（双路分离 -> 增强）
       -> 主说话人模型选择并门控
  -> 若不重叠：
       原音 -> 主说话人模型
       -> 高分：直接接受原音
       -> 低分：双路分离但不增强
          -> 主、辅两个说话人模型分别打分
          -> 两模型选中同一路、均过门限、均比原音改善时才救回
  -> 接受的音频 -> 最终 ASR -> 文本
  -> 拒绝的音频 -> 空文本
```

这套方法不靠简单降低总门限来减少误拒。低分样本必须通过双模型一致性与相对改善检查，避免为了召回正样本而无约束地放入负样本。原音始终保留；增强只用于重叠路径，非重叠救援不做增强，以减少说话人特征被改变的风险。

## 推荐的开源模型组合

| 功能 | 推荐项目 |
|---|---|
| 重叠检测 | [pyannote segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0) |
| 双人语音分离 | [TIGER](https://github.com/JusperLee/TIGER) |
| 语音增强 | [ClearerVoice-Studio 中的 MossFormerGAN](https://github.com/modelscope/ClearerVoice-Studio) |
| 唤醒候选转写 | [SenseVoice](https://github.com/QwenAudio/SenseVoice) |
| 主说话人验证 | [3D-Speaker 中的 CAM++](https://github.com/modelscope/3D-Speaker) |
| 救援辅助验证 | [3D-Speaker 中的 ERes2NetV2](https://github.com/modelscope/3D-Speaker/tree/main/egs/voxceleb/sv-eres2netv2) |
| 最终命令识别 | [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) |

仓库不捆绑这些项目。实际使用时，请从官方来源获取代码和权重，并分别核对其最新许可；部分模型托管页可能要求接受访问条件。

## 快速验证

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/mock_demo.py
python tools/prepublish_check.py
```

真实接入时，只需实现 `speaker_safe_pipeline.types` 中的五类小接口，再按照英文 README 的 API 示例组装 `ModelBundle`。

## 门限要求

`config.example.json` 故意不提供现成门限。门限应只用独立开发集校准，并覆盖真实设备、噪声、重叠、说话人和录音质量；最终测试集不能参与调参。评估时应同时报告误拒和误接收，不能只优化其中一项。

救援规则建议保持保守：两个说话人模型必须选中同一路；两个分数分别达到门限；两个分数都比未处理原音更高；并列不算一致。

## 许可边界

本仓库的组合逻辑使用 MIT License。第三方代码和模型权重仍遵循各自许可，本仓库不复制也不重新授权它们。
