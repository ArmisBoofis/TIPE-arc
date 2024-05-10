from colorama import Fore

# Function displaying a progress bar in the console
def progress_bar(partial_count, total_count, prefix=''):
    percentage = int(partial_count / float(total_count) * 100)
    filled_bar = '█' * percentage
    empty_bar = '-' * (100 - percentage)

    print(f'\r{prefix}|', end='')
    print(Fore.GREEN + filled_bar, end='') if partial_count == total_count else print(Fore.YELLOW + filled_bar, end='')
    print(Fore.RESET + empty_bar + f'| {percentage}% ({partial_count}/{total_count})', end='\r')