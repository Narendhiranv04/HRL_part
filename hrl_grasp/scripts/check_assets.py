from hrl_grasp.rlb_object_library import ensure_assets

if __name__ == '__main__':
    mapping = ensure_assets(headless=True)
    print('ASSET MAP:')
    for k in sorted(mapping.keys()):
        print(f'- {k}: {mapping[k]}')
