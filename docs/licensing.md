# Responsible-use license and third-party rights

The original IhsanVision.AI repository source code is published under the
[IhsanVision.AI Responsible Use License v1.0](../LICENSE). It permits
studying, running, adapting and sharing covered code, including legitimate
commercial software use, **only while complying with its conditions**.

**This is a use-restricted source-available license, not an OSI-approved
open-source license.** In particular, it prohibits knowingly or recklessly
causing or materially enabling harm. The full legally operative terms are
in [LICENSE](../LICENSE), not this summary.

## Summary of permissions

| Generally permitted, subject to the license | Not permitted under the license |
| --- | --- |
| Study and learn from the source | Fraud, falsified inspection results or deception |
| Reproduce and modify original project code | Weapons/violent harm or serious human-rights abuses |
| Build and distribute safe derivatives with attribution | Illegal surveillance, privacy abuse or discrimination |
| Use it in research or legitimate products | Reckless safety-critical deployment without proper safeguards |
| Legitimate commercial use of **this original source code** | Violating third-party dataset/weight/media licenses |

When distributing covered code, preserve the copyright and license notices,
identify modifications, and pass on the same use restrictions. Providing
remote access to a modified instance does not permit harmful use.
Good-faith authorized security research and lawful human-supervised
quality testing remain possible.

## Important: code and ML datasets have different licenses

**MVTec AD** and **MVTec AD 2** datasets are not licensed by this project.
MVTec states that the dataset is provided under
[CC BY-NC-SA 4.0](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
and that **commercial use of the dataset is not allowed without separate
permission**. The same dataset restriction is noted by
[MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2).

The project license therefore does not grant commercial rights to those
images, segmentation masks, other labels, checkpoints or other third-party
materials. **Commercial permission for original source code is not a
license for the MVTec data.** Any commercial implementation must obtain
lawful rights to its own image datasets, pretrained backbones, model weights,
dependencies and other third-party assets.

No MVTec dataset images or trained checkpoints are intentionally shipped in
this GitHub repository. The original project source license does not grant
a right to redistribute MVTec samples as demo screenshots, and no trained
checkpoint should be uploaded without an independent rights audit.

## Practical scope for the AI Engineer portfolio

- The demo and academic results remain historical research evidence, not a
  commercially validated or safety-certified product.
- Production users must audit false negatives and domain shift, implement
  human review, and validate their own safety and quality controls.
- Do not state that this repository uses MIT, Apache-2.0, GPL, or an
  OSI-approved open-source license; GitHub may label a custom license
  as “Other”.
- No license term can technically prevent misuse by itself. A rights holder
  may be able to seek legal remedies, but enforceability varies by
  jurisdiction, proof, copyright ownership and the precise facts. A
  qualified legal professional should review the custom wording before
  a high-stakes commercial distribution.
- Contributions and dependencies must not be relicensed contrary to rights
  the contributor or dependency owner actually controls.

This page is a plain-language summary, not legal advice and not a
replacement for the [full terms](../LICENSE).
