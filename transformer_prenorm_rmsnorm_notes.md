# Pre-Norm Transformer Blocks and RMSNorm

## English Explanation

### 1. What does a Transformer block do?

A Transformer block receives a tensor of token representations:

```python
x.shape == (batch_size, sequence_length, d_model)
```

Each token has a vector of length `d_model`. The block updates these vectors using two main sublayers:

1. **Multi-head self-attention**
   - Mixes information across tokens.
   - Example: the representation of token 5 can use information from tokens 1, 2, 3, 4, and 5.

2. **Feed-forward network**
   - Transforms each token vector independently.
   - It does not mix tokens; it only processes the hidden dimensions of each token.

The block output has the same shape as the input:

```python
(batch_size, sequence_length, d_model)
```

So a stack of Transformer blocks can repeatedly update the same residual stream.

---

### 2. What is a residual connection?

A residual connection means:

```text
output = input + change
```

For attention:

```python
x = x + attention(...)
```

For the feed-forward network:

```python
x = x + ffn(...)
```

The sublayer computes an update, but the original `x` is preserved and added back.

Intuition:

```text
Do not replace the representation completely.
Just add a useful correction to it.
```

This helps optimization because gradients can flow through the addition path even if the sublayer is hard to train.

---

### 3. Post-norm vs pre-norm

The original Transformer used **post-norm**:

```text
y = Norm(x + Attention(x))
z = Norm(y + FFN(y))
```

Meaning:

```text
first apply sublayer
then add residual
then normalize the result
```

Modern LLMs usually use **pre-norm**:

```text
y = x + Attention(Norm(x))
z = y + FFN(Norm(y))
```

Meaning:

```text
first normalize input
then apply sublayer
then add residual
```

Code shape:

```python
x = x + attention(norm1(x))
x = x + ffn(norm2(x))
```

The important difference is where normalization happens:

```text
post-norm: after residual addition
pre-norm: before sublayer computation
```

Pre-norm tends to train more stably in deep networks because the residual stream can pass through many layers without being normalized after every residual addition.

---

### 4. What is the “residual stream”?

Think of `x` as the main information highway through the Transformer.

Each layer adds updates to it:

```text
x0 -> x1 -> x2 -> x3 -> ... -> final x
```

With pre-norm:

```python
x_next = x + sublayer(norm(x))
```

The direct path from `x` to `x_next` is just addition. This makes it easier for information and gradients to move through the model.

---

### 5. What does RMSNorm do?

RMSNorm normalizes a vector by its magnitude.

For one token vector:

```text
a = [a1, a2, ..., ad]
```

Compute the root mean square:

```text
RMS(a) = sqrt(mean(a_i^2) + eps)
```

Then rescale each component:

```text
RMSNorm(a_i) = (a_i / RMS(a)) * g_i
```

`g_i` is a learned gain parameter. There is one gain value per hidden dimension:

```python
gain.shape == (d_model,)
```

So RMSNorm does two things:

1. Divide by vector magnitude, making the scale more stable.
2. Multiply by learned gain, letting the model choose useful per-dimension scales.

---

### 6. RMSNorm vs LayerNorm

LayerNorm:

```text
(x - mean) / std
```

RMSNorm:

```text
x / rms
```

Main difference:

```text
LayerNorm subtracts the mean.
RMSNorm does not subtract the mean.
```

RMSNorm only controls the vector magnitude. This is simpler and commonly used in modern LLMs.

---

### 7. Shape details

Input:

```python
x.shape == (batch_size, sequence_length, d_model)
```

RMSNorm computes the RMS over the final dimension:

```python
dim = -1
```

So for every token independently, it computes:

```text
sqrt(mean(hidden_vector^2) + eps)
```

The RMS value has shape:

```python
(batch_size, sequence_length, 1)
```

The gain has shape:

```python
(d_model,)
```

PyTorch broadcasting makes this work:

```python
result = normalized_x * gain
```

Output shape remains:

```python
(batch_size, sequence_length, d_model)
```

---

### 8. Why upcast to float32?

If `x` is `float16` or `bfloat16`, this operation can be numerically risky:

```python
x * x
```

because squaring can overflow or lose precision.

So RMSNorm should do:

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# compute RMSNorm safely
return result.to(in_dtype)
```

This keeps the internal math stable while returning the same dtype as the input.

---

## Türkçe Açıklama

### 1. Transformer block ne yapıyor?

Bir Transformer block, token temsil tensorunu alır:

```python
x.shape == (batch_size, sequence_length, d_model)
```

Her token için `d_model` uzunluğunda bir vektör vardır. Block bu vektörleri iki ana katmanla günceller:

1. **Multi-head self-attention**
   - Tokenlar arasında bilgi alışverişi yapar.
   - Örneğin 5. token, önceki tokenlardan bilgi alabilir.

2. **Feed-forward network**
   - Her token vektörünü bağımsız olarak dönüştürür.
   - Tokenlar arası bilgi karıştırmaz; sadece her tokenın hidden dimensionlarını işler.

Block input ve output shape’i aynıdır:

```python
(batch_size, sequence_length, d_model)
```

Bu yüzden birçok Transformer block arka arkaya dizilebilir.

---

### 2. Residual connection nedir?

Residual connection şu demektir:

```text
output = input + değişiklik
```

Attention için:

```python
x = x + attention(...)
```

Feed-forward için:

```python
x = x + ffn(...)
```

Yani katman `x`’i tamamen değiştirmez. Sadece `x` üzerine faydalı bir güncelleme ekler.

Sezgi:

```text
Temsili sıfırdan yeniden yazma.
Mevcut temsilin üstüne düzeltme ekle.
```

Bu eğitimde yardımcı olur çünkü gradientler residual ekleme yolu üzerinden daha rahat akar.

---

### 3. Post-norm ve pre-norm farkı

Orijinal Transformer **post-norm** kullanıyordu:

```text
y = Norm(x + Attention(x))
z = Norm(y + FFN(y))
```

Yani:

```text
önce katmanı uygula
sonra residual ekle
sonra normalize et
```

Modern LLM’ler genellikle **pre-norm** kullanır:

```text
y = x + Attention(Norm(x))
z = y + FFN(Norm(y))
```

Yani:

```text
önce normalize et
sonra katmanı uygula
sonra residual ekle
```

Kod olarak:

```python
x = x + attention(norm1(x))
x = x + ffn(norm2(x))
```

Ana fark:

```text
post-norm: normalization residual eklemeden sonra
pre-norm: normalization sublayer'dan önce
```

Pre-norm derin modellerde daha stabil eğitilir çünkü residual stream her katmandan sonra normalize edilmeden ilerleyebilir.

---

### 4. Residual stream nedir?

`x`’i modelin ana bilgi yolu gibi düşünebilirsin.

Her layer bu yola bir güncelleme ekler:

```text
x0 -> x1 -> x2 -> x3 -> ... -> final x
```

Pre-norm’da:

```python
x_next = x + sublayer(norm(x))
```

Burada `x` doğrudan `x_next`’e eklenerek geçer. Bu, bilginin ve gradientlerin derin model içinde daha rahat taşınmasına yardımcı olur.

---

### 5. RMSNorm ne yapıyor?

RMSNorm bir vektörü büyüklüğüne göre normalize eder.

Bir token vektörü:

```text
a = [a1, a2, ..., ad]
```

Önce root mean square hesaplanır:

```text
RMS(a) = sqrt(mean(a_i^2) + eps)
```

Sonra her eleman şöyle ölçeklenir:

```text
RMSNorm(a_i) = (a_i / RMS(a)) * g_i
```

`g_i` öğrenilen gain parametresidir. Her hidden dimension için bir tane vardır:

```python
gain.shape == (d_model,)
```

Yani RMSNorm iki şey yapar:

1. Vektörü büyüklüğüne böler, scale’i stabilize eder.
2. Öğrenilen gain ile her dimension’ı tekrar uygun şekilde ölçekler.

---

### 6. RMSNorm vs LayerNorm

LayerNorm:

```text
(x - mean) / std
```

RMSNorm:

```text
x / rms
```

En büyük fark:

```text
LayerNorm mean çıkarır.
RMSNorm mean çıkarmaz.
```

RMSNorm sadece vektörün büyüklüğünü kontrol eder. Daha basittir ve modern LLM’lerde yaygındır.

---

### 7. Shape mantığı

Input:

```python
x.shape == (batch_size, sequence_length, d_model)
```

RMSNorm son dimension üzerinden hesaplanır:

```python
dim = -1
```

Yani her token için ayrı ayrı:

```text
sqrt(mean(hidden_vector^2) + eps)
```

hesaplanır.

RMS değeri shape olarak:

```python
(batch_size, sequence_length, 1)
```

Gain shape’i:

```python
(d_model,)
```

PyTorch broadcasting sayesinde:

```python
result = normalized_x * gain
```

doğru çalışır.

Output shape değişmez:

```python
(batch_size, sequence_length, d_model)
```

---

### 8. Neden float32’ye upcast ediyoruz?

Eğer `x` `float16` veya `bfloat16` ise şu işlem riskli olabilir:

```python
x * x
```

Çünkü kare alma overflow’a veya precision kaybına yol açabilir.

Bu yüzden RMSNorm’da genelde:

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# RMSNorm hesapla
return result.to(in_dtype)
```

yapılır.

Böylece iç hesap daha güvenli olur, ama çıktı yine input ile aynı dtype’a döner.
