# Position-Wise Feed-Forward Network, SiLU, GLU, and SwiGLU

## FFN Nedir?

Transformer block içinde iki ana parça vardır:

```text
1. Attention: tokenlar birbirleriyle konuşur
2. FFN / MLP: her token kendi içinde işlenir
```

**Position-wise Feed-Forward Network** demek, aynı küçük MLP'nin her token pozisyonuna ayrı ayrı uygulanması demektir.

Input shape genelde şöyledir:

```text
x: batch_size sequence_length d_model
```

FFN, `sequence_length` boyunca tokenları birbirine karıştırmaz. Her token vektörüne aynı dönüşümü uygular:

```text
batch sequence d_model -> batch sequence d_model
```

Tokenlar arası bilgi karışımı attention'da olur. FFN ise her token'ın feature'larını kendi içinde daha zengin hale getirir.

---

## Original Transformer FFN

Attention Is All You Need paper'ındaki FFN şuydu:

```text
FFN(x) = W2 ReLU(W1 x)
```

Shape flow:

```text
x:          d_model
W1 x:       d_ff
ReLU:       d_ff
W2 (...):   d_model
```

Yani önce hidden dimension büyütülür, nonlinearity uygulanır, sonra tekrar `d_model` boyutuna dönülür.

Classic Transformer'da genelde:

```text
d_ff = 4 * d_model
```

Örneğin:

```text
d_model = 512
d_ff    = 2048
```

---

## ReLU Nedir?

ReLU en basit activation function'lardan biridir:

```text
ReLU(x) = max(0, x)
```

Yani:

```text
x negatifse -> 0
x pozitifse -> x
```

Problem: zero civarında keskindir ve negatif tarafı tamamen öldürür.

---

## SiLU / Swish Nedir?

Modern LLM'lerde ReLU yerine çoğunlukla daha smooth activation'lar kullanılır. Bunlardan biri **SiLU**, diğer adıyla **Swish**:

```text
SiLU(x) = x * sigmoid(x)
```

Sigmoid:

```text
sigmoid(x) = 1 / (1 + e^(-x))
```

Dolayısıyla:

```text
SiLU(x) = x / (1 + e^(-x))
```

PyTorch implementation:

```python
def silu(x):
    return x * torch.sigmoid(x)
```

ReLU'ya göre farkı:

- zero civarında smooth'tur
- negatif değerleri tamamen sıfırlamaz
- gradient davranışı daha yumuşaktır

Assignment özellikle `torch.sigmoid` kullanmaya izin veriyor çünkü numerically stable implementation PyTorch tarafında hazırdır.

---

## GLU / Gating Nedir?

GLU, **Gated Linear Unit** demektir. Ana fikir:

> Bir branch, diğer branch'teki feature'ların ne kadar geçeceğini kontrol eder.

Classic GLU formu:

```text
GLU(x, W1, W2) = sigmoid(W1 x) ⊙ W2 x
```

Buradaki `⊙` element-wise multiplication demektir.

Mental model:

```text
gate  = sigmoid(W1 x)   # ne kadar aç/kapat?
value = W2 x            # taşınacak bilgi
out   = gate * value
```

Yani gate değeri küçükse bazı feature'lar bastırılır; büyükse geçmesine izin verilir.

---

## SwiGLU Nedir?

**SwiGLU**, SiLU/Swish activation ile GLU gating fikrinin birleşimidir.

Assignment'taki modern FFN formu:

```text
FFN(x) = W2( SiLU(W1 x) ⊙ W3 x )
```

Burada 3 linear projection vardır:

```text
W1: d_model -> d_ff
W3: d_model -> d_ff
W2: d_ff    -> d_model
```

Forward flow:

```text
a = SiLU(W1 x)
b = W3 x
h = a ⊙ b
out = W2 h
```

Kod olarak concept:

```python
hidden = silu(w1(x)) * w3(x)
out = w2(hidden)
```

Burada:

- `w1(x)` gate-like nonlinear branch üretir
- `w3(x)` value/content branch üretir
- element-wise multiplication ile feature'lar modüle edilir
- `w2` sonucu tekrar `d_model` boyutuna indirir

---

## Neden SwiGLU Kullanıyoruz?

Modern language model'larda SwiGLU, plain ReLU FFN'e göre daha iyi sonuç verebiliyor.

Pratik sebepler:

- gating sayesinde feature seçimi daha esnek olur
- SiLU smooth olduğu için optimization daha iyi davranabilir
- LLaMA, Qwen gibi modern LLM mimarilerinde yaygın olarak kullanılır

Ama önemli nokta: Bu tamamen teorik olarak “kanıtlanmış tek doğru yol” değil. Shazeer'ın meşhur alıntısı da bunu güzel özetler:

> “We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.”

Yani: çalıştığı empirically görülmüş, bu yüzden modern LLM'lerde tercih ediliyor.

---

## Neden `d_ff = 8/3 * d_model`?

Classic FFN'de iki matrix vardı:

```text
W1: d_model -> 4d_model
W2: 4d_model -> d_model
```

Parametre sayısı yaklaşık:

```text
4d² + 4d² = 8d²
```

SwiGLU'da üç matrix var:

```text
W1: d_model -> d_ff
W3: d_model -> d_ff
W2: d_ff    -> d_model
```

Parametre sayısı yaklaşık:

```text
d*d_ff + d*d_ff + d_ff*d = 3d*d_ff
```

Classic FFN ile benzer parametre bütçesi için:

```text
3d*d_ff ≈ 8d²
d_ff ≈ 8/3 * d
```

Bu yüzden assignment şunu istiyor:

```text
d_ff ≈ 8/3 * d_model
```

---

## Neden 64'ün Katına Yuvarlıyoruz?

Matmul operasyonları GPU/accelerator üzerinde belirli boyutlarda daha verimli çalışır. Bu yüzden inner dimension genelde 64'ün katı yapılır.

Basit yaklaşım:

```python
d_ff = int(8 * d_model / 3)
d_ff = 64 * math.ceil(d_ff / 64)
```

Örnek:

```text
d_model = 512
8/3 * 512 = 1365.33
64'e yukarı yuvarla -> 1408
```

---

## Shape Flow Örneği

Input:

```text
x: batch sequence d_model
```

`W1 x`:

```text
batch sequence d_ff
```

`SiLU(W1 x)`:

```text
batch sequence d_ff
```

`W3 x`:

```text
batch sequence d_ff
```

Element-wise multiply:

```text
batch sequence d_ff
```

`W2(...)`:

```text
batch sequence d_model
```

Final output input ile aynı shape'tedir:

```text
batch sequence d_model
```

Bu yüzden Transformer block içinde residual connection ile rahatça toplanabilir:

```text
x = x + FFN(norm(x))
```

---

## Assignment İçin Implementation Mental Model

Senin mevcut `Linear` class'ınla FFN kabaca şöyle düşünülür:

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = 64 * math.ceil(d_ff / 64)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

    def silu(self, x):
        return x * torch.sigmoid(x)
```

Önemli noktalar:

- `nn.Linear` kullanma; assignment kendi `Linear` implementation'ını istiyor.
- Bias yok.
- `W1` ve `W3` output shape'i aynı olmalı: `d_ff`.
- Çarpım element-wise yapılır.
- Final output tekrar `d_model` olmalı.
