import torch

from word2eq.decoding import GrammarTables, greedy_decode
from word2eq.expr import OPERATORS, is_valid_prefix, number_index
from word2eq.model import ModelConfig, Seq2SeqTransformer
from word2eq.vocab import Vocab, number_tokens


def _random_model(seed: int, tgt: Vocab) -> Seq2SeqTransformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(src_vocab=50, tgt_vocab=len(tgt), d_model=32, nhead=2, enc_layers=1, dec_layers=1,
                      ff_dim=64, dropout=0.0)
    return Seq2SeqTransformer(cfg).eval()


def test_constrained_decoding_is_always_valid_even_for_untrained_models():
    tgt = Vocab.build([], reserved=[*OPERATORS, *number_tokens(), "100.0", "0.01"])
    grammar = GrammarTables.from_vocab(tgt)
    for seed in range(20):
        model = _random_model(seed, tgt)
        g = torch.Generator().manual_seed(seed)
        src = torch.randint(4, 50, (16, 12), generator=g)
        n_nums = torch.randint(1, 5, (16,), generator=g)
        max_len = int(torch.randint(1, 12, (1,), generator=g))
        out = greedy_decode(model, src, n_nums, grammar, max_len=max_len)
        for row, n in zip(out.tolist(), n_nums.tolist(), strict=True):
            toks = tgt.decode(row)
            assert is_valid_prefix(toks), toks
            assert len(toks) <= max_len
            # only references numbers the problem actually contains
            assert all((number_index(t) or 0) < n for t in toks)


def test_unconstrained_random_model_produces_invalid_output():
    tgt = Vocab.build([], reserved=[*OPERATORS, *number_tokens()])
    model = _random_model(0, tgt)
    out = greedy_decode(model, torch.randint(4, 50, (32, 10)), torch.full((32,), 2), None, max_len=8)
    assert not all(is_valid_prefix(tgt.decode(r)) for r in out.tolist())
