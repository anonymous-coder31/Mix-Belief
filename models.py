from typing import Dict, Iterable, Optional, Set, cast

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
)


class MyBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return cast(nn.Embedding, self.embeddings.word_embeddings)

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                f"set_input_embeddings expects nn.Embedding, got {type(value).__name__}"
            )
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune: Dict[int, Iterable[int]]) -> None:
        """
        heads_to_prune : { layer_index: iterable_of_head_indices }
        Exemple : { 0: [0, 1, 2], 5: {3, 7} }
        """
        for layer_idx, heads in heads_to_prune.items():
            # 1) Normaliser en set[int]
            if isinstance(heads, torch.Tensor):
                # aplatit puis cast en int natif
                heads_norm: Set[int] = set(int(h) for h in heads.flatten().tolist())
            else:
                heads_norm = set(int(h) for h in heads)

            # 2) Récupérer le bloc d'attention
            attn = self.encoder.layer[layer_idx].attention

            # 3) Selon les versions, prune_heads est sur .self (BertSelfAttention) ou sur le wrapper (BertAttention)
            target = getattr(attn, "self", attn)

            # 4) Vérifier que prune_heads est bien appelable
            fn = getattr(target, "prune_heads", None)
            if fn is None or not callable(fn):
                raise AttributeError(
                    "Attention module has no callable 'prune_heads'. "
                    "Vérifie que tu utilises un modèle BERT/HF compatible."
                )

            # 5) Appeler avec un set d'indices (format attendu par HF)
            fn(heads_norm)

    def _forward_init(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if input_ids is not None:
            device = input_ids.device
        elif inputs_embeds is not None:
            device = inputs_embeds.device
        else:
            device = next(self.parameters()).device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        return (
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            extended_attention_mask,
            encoder_extended_attention_mask,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else getattr(self.config, "output_attentions", False)
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "use_return_dict", False)
        )
        (
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            extended_attention_mask,
            encoder_extended_attention_mask,
        ) = self._forward_init(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
        )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            sequence_output = encoder_outputs.last_hidden_state
            pooled_output = self.pooler(sequence_output)

            return {
                "last_hidden_state": sequence_output,
                "pooler_output": pooled_output,
                "hidden_states": encoder_outputs.hidden_states,
                "attentions": encoder_outputs.attentions,
            }
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def forward_mix_embed(self, x1, att1, x2, att2, lam):
        (
            x1,
            attention_mask1,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            extended_attention_mask1,
            encoder_extended_attention_mask,
        ) = self._forward_init(input_ids=x1, attention_mask=att1)
        embedding_output1 = self.embeddings(
            input_ids=x1,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        (
            x2,
            attention_mask2,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            extended_attention_mask2,
            encoder_extended_attention_mask,
        ) = self._forward_init(input_ids=x2, attention_mask=att2)

        embedding_output2 = self.embeddings(
            input_ids=x2,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        embedding_output = lam * embedding_output1 + (1.0 - lam) * embedding_output2

        # need to take max of both to ensure we don't miss attending to any value
        extended_attention_mask = torch.max(
            extended_attention_mask1, extended_attention_mask2
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output, embedding_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    # inverse de softplus (approx exacte): softplus(x)=log(1+exp(x))
    # inv_softplus(y)=log(exp(y)-1)
    return torch.log(torch.expm1(y))


class TextBERT(nn.Module):
    def __init__(
        self,
        pretrained_model,
        num_class,
        fine_tune,
        dropout,
        freeze,
        use_unc=False,
    ):
        super(TextBERT, self).__init__()
        self.output_dim = num_class
        self.bert = MyBertModel.from_pretrained(pretrained_model)

        # Freeze bert layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:freeze]:  # Gel the first 6 layers
            for param in layer.parameters():
                param.requires_grad = False

        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        # bert_dim = 1024
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_class)

        # ---- Balanced-EDL prior parameter (learnable) ----
        # init pour que beta = softplus(beta_raw) ~= 1 (comportement vanilla Dir(1) au départ)
        beta0 = torch.ones(num_class)  # Dir(1)
        beta_raw_init = inv_softplus(beta0)

        if use_unc:
            self.beta_raw = nn.Parameter(beta_raw_init)  # learnable [C]
        else:
            self.register_buffer("beta_raw", beta_raw_init)  # non-learnable [C]

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=False,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        if return_dict:
            pooled_output = outputs["pooler_output"]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            return {
                "logits": logits,
                "pooled_output": pooled_output,
                "last_hidden_state": outputs["last_hidden_state"],
                "hidden_states": outputs["hidden_states"],
                "attentions": outputs["attentions"],
            }

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output

    def forward_mix_embed(self, x1, att1, x2, att2, lam):
        outputs = self.bert.forward_mix_embed(x1, att1, x2, att2, lam)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def forward_mix_sent(self, x1, att1, x2, att2, lam):
        logits1 = self.forward(x1, att1)
        logits2 = self.forward(x2, att2)
        y = lam * logits1 + (1.0 - lam) * logits2
        return y

    def forward_mix_encoder(self, x1, att1, x2, att2, lam):
        outputs1 = self.bert(
            input_ids=x1,
            attention_mask=att1,
            return_dict=False,
        )

        outputs2 = self.bert(
            input_ids=x2,
            attention_mask=att2,
            return_dict=False,
        )
        pooled_output1 = self.dropout(outputs1[1])
        pooled_output2 = self.dropout(outputs2[1])
        pooled_output = lam * pooled_output1 + (1.0 - lam) * pooled_output2
        logits_org = self.classifier(pooled_output1)
        logits_mix = self.classifier(pooled_output)
        return logits_mix
