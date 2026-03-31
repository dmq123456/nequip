import logging
from nequip.nn import GraphModuleMixin
import torch


def tune_model_coeff_nosoc(model: GraphModuleMixin, state: int):
    """
    Tune state of nosoc model
    E = c_r * E_r + c_hb * E_hb + c_nosoc * E_nosoc (or + c_soc * E_soc)
    State:
        0: only E_r (c_hb = c_nosoc = 0)
        1: Frozen E_r and E_hb (E_r of False requiregrads and c_nosoc = 0)
        2: E_r and E_hb (c_nosoc = 0)
        3: Frozen E_r, E_hb and E_nosoc (E_r and E_hb of False requiregrads)
        4: E_r, E_hb and E_nosoc
        5: Frozen E_r, E_hb, E_nosoc and E_soc (E_r, E_hb, E_nosoc of False requiregrads)
        6: E_r, E_hb, E_nosoc and E_soc
    """

    def _freeze_mod(mod: torch.nn.Module):
        for param in mod.parameters():
            param.requires_grad = False

    def _activate_mod(mod: torch.nn.Module):
        for param in mod.parameters():
            param.requires_grad = True

    def _tune_model_coeff_nosoc(mod: torch.nn.Module):
        # Ensure turn off nosoc/soc coeff_trainable
        if hasattr(mod, "layer_name"):
            if state == 0:
                logging.info("State 0: only E_r")
                if "edge_onlyr_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_hb_eng" in mod.layer_name:
                    mod.coeff.zero_()
                elif "edge_eng_nosoc" in mod.layer_name:
                    mod.coeff.zero_()
                elif "edge_eng_soc_on" in mod.layer_name:
                    mod.coeff.zero_()

                if "_onlyr" in mod.layer_name:
                    _activate_mod(mod)
                elif "_hb" in mod.layer_name:
                    _freeze_mod(mod)
                elif "_nosoc" in mod.layer_name:
                    _freeze_mod(mod)
                elif "_soc_on" in mod.layer_name:
                    _freeze_mod(mod)
            elif state == 1:
                logging.info("State 1: Frozen E_r + E_hb")
                if "edge_onlyr_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_hb_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_eng_nosoc" in mod.layer_name:
                    mod.coeff.zero_()
                elif "edge_eng_soc_on" in mod.layer_name:
                    mod.coeff.zero_()

                if "_onlyr" in mod.layer_name:
                    _freeze_mod(mod)
                elif "_hb" in mod.layer_name:
                    _activate_mod(mod)
                elif "_nosoc" in mod.layer_name:
                    _freeze_mod(mod)
                elif "_soc_on" in mod.layer_name:
                    _freeze_mod(mod)
            elif state == 2:
                logging.info("State 2: E_r + E_hb")
                if "edge_onlyr_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_hb_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_eng_nosoc" in mod.layer_name:
                    mod.coeff.zero_()
                elif "edge_eng_soc_on" in mod.layer_name:
                    mod.coeff.zero_()

                if "_onlyr" in mod.layer_name:
                    _activate_mod(mod)
                elif "_hb" in mod.layer_name:
                    _activate_mod(mod)
                elif "_nosoc" in mod.layer_name:
                    _freeze_mod(mod)
                elif "_soc_on" in mod.layer_name:
                    _freeze_mod(mod)
            elif state == 3:
                logging.info("State 3: Frozen E_r + E_hb and E_nosoc")
                if "edge_onlyr_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_hb_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_eng_nosoc" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_eng_soc_on" in mod.layer_name:
                    mod.coeff.zero_()

                if "_onlyr" in mod.layer_name:
                    _freeze_mod(mod)
                elif "_hb" in mod.layer_name:
                    _freeze_mod(mod)
                elif "_nosoc" in mod.layer_name:
                    _activate_mod(mod)
                elif "_soc_on" in mod.layer_name:
                    _freeze_mod(mod)
            elif state == 4:
                logging.info("State 4: E_r + E_hb + E_nosoc")
                if "edge_onlyr_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_hb_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_eng_nosoc" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_eng_soc_on" in mod.layer_name:
                    mod.coeff = mod.zero_()

                if "_onlyr" in mod.layer_name:
                    _activate_mod(mod)
                elif "_hb" in mod.layer_name:
                    _activate_mod(mod)
                elif "_nosoc" in mod.layer_name:
                    _activate_mod(mod)
                elif "_soc_on" in mod.layer_name:
                    _freeze_mod(mod)
            elif state == 5:
                logging.info("State 5: Fronze E_r + E_hb + E_nosoc, and E_soc_on")
                if "edge_onlyr_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_hb_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_eng_nosoc" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_eng_soc_on" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)

                if "_onlyr" in mod.layer_name:
                    _freeze_mod(mod)
                elif "_hb" in mod.layer_name:
                    _freeze_mod(mod)
                elif "_nosoc" in mod.layer_name:
                    _freeze_mod(mod)
                elif "_soc_on" in mod.layer_name:
                    _activate_mod(mod)
            elif state == 6:
                logging.info("State 6: E_r + E_hb + E_nosoc, and E_soc_on")
                if "edge_onlyr_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_hb_eng" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_eng_nosoc" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)
                elif "edge_eng_soc_on" in mod.layer_name:
                    mod.coeff.fill_(mod.coeff_og)

                if "_onlyr" in mod.layer_name:
                    _activate_mod(mod)
                elif "_hb" in mod.layer_name:
                    _activate_mod(mod)
                elif "_nosoc" in mod.layer_name:
                    _activate_mod(mod)
                elif "_soc_on" in mod.layer_name:
                    _activate_mod(mod)
            else:
                raise ValueError("Wrong state {}".format(state))
        pass

    with torch.no_grad():
        model.apply(_tune_model_coeff_nosoc)
    return model


def tune_model_ztype(model: GraphModuleMixin, state: int):
    """
    Tune state of nosoc model
    E = E_ztype + E_nn
    State:
        0: only E_ztype
        1: Frozen E_ztype and train E_nn (E_ztype of False requiregrads)
        2: E_ztype and E_nn
    """

    def _freeze_mod(mod: torch.nn.Module):
        for param in mod.parameters():
            param.requires_grad = False

    def _activate_mod(mod: torch.nn.Module):
        for param in mod.parameters():
            param.requires_grad = True

    def _tune_model_ztype(mod: torch.nn.Module):
        # Ensure turn off nosoc/soc coeff_trainable
        if hasattr(mod, "layer_name"):
            print(mod.layer_name)
            if state == 0:
                if "species_eng" in mod.layer_name:
                    logging.info("State: 0, only E_ztype")
                    mod.add_collect = False
                    _activate_mod(mod)
            elif state == 1:
                if "species_eng" in mod.layer_name:
                    logging.info("State: 1, only Frozen E_ztype and train E_nn")
                    mod.add_collect = True
                    _freeze_mod(mod)
            elif state == 2:
                if "species_eng" in mod.layer_name:
                    logging.info("State: 2, E_ztype and E_nn")
                    mod.add_collect = True
                    _activate_mod(mod)
            else:
                raise ValueError("Wrong state {}".format(state))

    with torch.no_grad():
        model.apply(_tune_model_ztype)
    return model
